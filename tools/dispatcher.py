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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

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

    return spec.fn(principal=principal, run_id=run_id, **kwargs)
