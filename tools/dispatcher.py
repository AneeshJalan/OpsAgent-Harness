"""The dispatch boundary. A registry (registry_c.REGISTRY_C or registry_s.REGISTRY_S) is a
plain `{tool_name: ToolSpec}` dict — a persona's dispatcher is built from exactly one of them,
so a tool absent from that dict cannot be called against that persona at all. That's the first
enforcement layer (stops accidents: the tool was never wired in).

`dispatch()` is the second layer (stops attacks): even given a registry and a tool name that's
a legitimate key in it, this still re-validates before calling anything —

  1. Tool not in the given registry -> DENIED, reason=not_in_registry, logged directly here
     (the tool function is never reached, so nothing else would log this call at all).
  2. Tool exists but the principal's role doesn't meet its `min_role` gate (e.g.
     record_payment, manager+) -> DENIED, reason=insufficient_role, logged directly here.
  3. Otherwise, the tool function itself is called with the principal it will actually run
     as.

What does NOT live here: ownership scoping (does this invoice belong to this customer?).
That check needs domain knowledge — which argument names what, and what a lookup means for
this specific tool — that a generic dispatcher can't have without effectively reimplementing
every tool's logic here too. So each ownership-scoped tool re-checks the principal against the
record it's about to touch, as literally the first thing it does, and logs
`reason='principal_mismatch'` itself before returning DENIED without touching any state. The
guarantee dispatch() provides is that this always happens *before* execution, never after.

One more thing that does live here: datetime coercion. JSON has no datetime type, so a
tool_use.input a model actually sends carries every datetime argument as an ISO-8601 string
(agent/schemas.py declares them exactly that way) — but every tool function's own signature
types those parameters as real `datetime` objects and does arithmetic on them directly
(`start_ts + timedelta(...)`). Every caller of dispatch() (today: the agent loop; potentially
more in the future) would otherwise have to remember to convert those strings itself before
calling in, and every existing test happens to already pass real datetime objects — so this
gap was invisible until the agent loop's own tests exercised it. Coercing once, here, at the one
boundary every caller goes through, is both DRY and the only way to guarantee it never regresses
regardless of what calls dispatch() next.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, get_type_hints

from db.database import get_session
from tools.audit import write_audit
from tools.principal import Principal, StaffRole
from tools.reasons import Reason


class Decision(str, Enum):
    EXECUTED = "executed"
    NEEDS_CONFIRM = "needs_confirm"
    QUEUED = "queued"
    DENIED = "denied"


@dataclass(frozen=True)
class ToolSpec:
    fn: Callable[..., dict[str, Any]]
    tier: int  # declared tier, 0-3 — static metadata, not the runtime decision
    min_role: StaffRole | None = None  # staff-only; None = no role gate beyond registry membership


Registry = dict[str, ToolSpec]


def _coerce_datetime_args(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Replaces any ISO-8601 string argument with a real `datetime`, wherever `fn`'s own
    signature types that parameter as `datetime`. Registry modules use `from __future__ import
    annotations`, which turns every annotation into an unevaluated string at runtime --
    get_type_hints() resolves those against the tool function's own module globals, the same
    approach agent/schemas.py already uses to build strict JSON schemas from these signatures.
    A malformed string is left for the tool call's own exception handling to surface (the agent
    loop turns a raised exception into a tool_result with is_error: true, never a crash).

    A tz-aware string has its offset **discarded, not converted** -- the wall-clock reading the
    model wrote is taken as the intended local business time. Every timestamp in this system is
    naive and means local time at the business: `business_hours` is 08:00-18:00 local, seeded
    appointment rows are local, and `db.seed_common.now_utc()` is the same naive frame. Nothing
    here is genuinely UTC despite that function's name, and the business operates in exactly one
    timezone, so there is no second frame to convert between.

    Converting instead of discarding is what this used to do, and it silently moved every time
    the model qualified with a real offset. Claude includes *some* offset almost every time and
    picks it inconsistently -- one eval run carried `Z`, `-00:00`, `-04:00` and `-07:00` across
    different cases of the same suite. `Z`/`-00:00` happened to be no-ops; `-07:00` shifted
    every value seven hours, so a caller asking for a 10:00 appointment had it checked, booked
    and reported as 17:00. Availability came back for the wrong window, business-hours checks
    ran against the wrong hour, and the agent told callers their requested time was unavailable.
    Discarding the offset makes the coercion deterministic regardless of which timezone the
    model guesses.

    Stripping tzinfo at all is still required, in either direction: comparing a tz-aware value
    against a naive one raises TypeError rather than producing a graceful denial, so this is the
    one boundary that normalizes, instead of every comparison site downstream (`tools/policy.py`,
    `get_availability`, ...) having to handle both cases."""
    hints = get_type_hints(fn)
    coerced = dict(kwargs)
    for name, value in kwargs.items():
        if isinstance(value, str) and hints.get(name) is datetime:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is not None:
                parsed = parsed.replace(tzinfo=None)
            coerced[name] = parsed
    return coerced


def _log_pre_dispatch_denial(
    *, principal: Principal, tool_name: str, tier: int, reason: str, run_id: str | None, args: dict[str, Any]
) -> None:
    """Used only for denials that happen before any tool function runs — the tool's own
    session was never opened, so this is the one place that opens a short-lived session just
    to record that the call was rejected."""
    with get_session() as session:
        write_audit(
            session,
            principal=principal,
            tool=tool_name,
            declared_tier=tier,
            decision=Decision.DENIED.value,
            args=args,
            reason=reason,
            run_id=run_id,
        )
        session.commit()


def dispatch(
    registry: Registry,
    tool_name: str,
    principal: Principal,
    *,
    run_id: str | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    spec = registry.get(tool_name)
    if spec is None:
        _log_pre_dispatch_denial(
            principal=principal, tool_name=tool_name, tier=0,
            reason=Reason.NOT_IN_REGISTRY.value, run_id=run_id, args=kwargs,
        )
        return {"decision": Decision.DENIED.value, "reason": Reason.NOT_IN_REGISTRY.value, "tool": tool_name}

    if spec.min_role is not None and not principal.has_role_at_least(spec.min_role):
        _log_pre_dispatch_denial(
            principal=principal, tool_name=tool_name, tier=spec.tier,
            reason=Reason.INSUFFICIENT_ROLE.value, run_id=run_id, args=kwargs,
        )
        return {"decision": Decision.DENIED.value, "reason": Reason.INSUFFICIENT_ROLE.value, "tool": tool_name}

    kwargs = _coerce_datetime_args(spec.fn, kwargs)
    return spec.fn(principal=principal, run_id=run_id, **kwargs)
