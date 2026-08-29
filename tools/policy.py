"""The business's operating envelope, read from policy_config, plus the individual checks
that decide whether a request stays inside it. Each check returns (ok, reason_code | None) —
turning a failed check into QUEUED, NEEDS_CONFIRM, or DENIED is the calling tool's job, not
this module's. Keeping that split means the same check can back different runtime decisions
in different tools (e.g. a failed business-hours check queues a customer booking but is simply
overridable by staff via book_appointment_for_customer).

## The policy-in-prompt vs. policy-in-code ablation (Planning/DAY3.md §2.2)

`POLICY_ENFORCEMENT` is a harness-only escape hatch, read exactly once in this file (see
`_envelope_enforced` below), that lets `check_business_hours`, `check_lead_time`,
`check_booking_window`, and `check_balance_hold` short-circuit to "always passes" instead of
their real logic. Its only purpose is to make the ablation in DAY3 §3.1 ("policy stated in the
prompt vs. enforced only in code") an actually-true test of that hypothesis, rather than one
that measures nothing because the code-level backstop is silently still there under a
prompt-only-labeled run.

**Default is `"code"` — fully enforced, same as if this section didn't exist.** The env var is
never read by, set from, or reachable through a tool argument, a registry entry, or anything the
model can see; the only writer is `evals/case_runner.py`'s own ablation wiring, which sets it for
the duration of exactly one case run when that case is using the `policy_in_prompt` prompt
variant (`agent/prompts.py`'s `SYSTEM_C_POLICY_IN_PROMPT`), and restores whatever was there
before immediately afterward. It is never left set across cases and never defaults to anything
but full enforcement.

**Scope is deliberately narrow — four checks, not the whole envelope.** Only the checks whose
content is actually restated as prose in `SYSTEM_C_POLICY_IN_PROMPT` (business hours, lead time,
booking window, balance hold) are gated. `check_discount` is staff-only and was never in that
prompt. `deposit_required` and `cancellation_fee_applies` are in-conversation confirmation gates
the model must relay honestly (a `NEEDS_CONFIRM`, not a silent auto-`QUEUED`) — ablating
"stop auto-escalating" doesn't describe them, so they're untouched. And the online-bookable,
auto-book-enabled, and skilled-technician checks are catalog/operational facts never described
to the model at all either way; bypassing them would just corrupt bookings (e.g. one with no
technician assigned) rather than test anything about prompt-vs-code policy.

**A latent, accepted coupling:** these same four check functions are also called from
`tools/registry_s.py`'s `book_appointment_for_customer` (staff). This switch is never actually
live during a Persona-S call in practice — `evals/case_runner.py`'s `SYSTEM_PROMPTS` table has
no `"S"` entry under `policy_in_prompt`, so an S-persona case run under that variant fails to
even select a system prompt before any tool is ever dispatched — but it's worth knowing this
module doesn't itself distinguish the two registries.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from sqlalchemy.orm import Session

from db.models import PolicyConfig
from tools.reasons import Reason

# ABLATION ONLY -- see the module docstring above. Never default (absent or any other value
# means fully enforced). Never reachable from a tool argument, a registry entry, or any
# model-visible surface -- only evals/case_runner.py's ablation wiring ever sets this.
_POLICY_ENFORCEMENT_ENV_VAR = "POLICY_ENFORCEMENT"


def _envelope_enforced() -> bool:
    """The one and only read of _POLICY_ENFORCEMENT_ENV_VAR in this codebase."""
    return os.environ.get(_POLICY_ENFORCEMENT_ENV_VAR, "code") != "prompt_only"


@dataclass(frozen=True)
class Policy:
    business_hours: dict  # {"mon_fri": {"open","close"}|None, "sat": ..., "sun": ...}
    min_lead_time_hours: int
    max_booking_window_days: int
    auto_book_enabled: bool
    deposit_required_above: int  # cents
    blocking_balance_above: int  # cents
    max_discount_pct: int
    cancellation_fee_window_hrs: int
    after_hours_booking: str  # never | deferred | allowed
    identity_mode: str  # weak | strong


def _to_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_policy(session: Session) -> Policy:
    """Reads policy_config fresh on every call — v1 scope is a handful of rows read at most
    once per tool call, so a cache would be optimizing a cost that doesn't exist yet."""
    rows = {row.key: row.value for row in session.query(PolicyConfig).all()}
    return Policy(
        business_hours=json.loads(rows["business_hours"]),
        min_lead_time_hours=int(rows["min_lead_time_hours"]),
        max_booking_window_days=int(rows["max_booking_window_days"]),
        auto_book_enabled=_to_bool(rows["auto_book_enabled"]),
        deposit_required_above=int(rows["deposit_required_above"]),
        blocking_balance_above=int(rows["blocking_balance_above"]),
        max_discount_pct=int(rows["max_discount_pct"]),
        cancellation_fee_window_hrs=int(rows["cancellation_fee_window_hrs"]),
        after_hours_booking=rows["after_hours_booking"],
        identity_mode=rows["identity_mode"],
    )


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def check_business_hours(policy: Policy, start_ts: datetime) -> tuple[bool, str | None]:
    """`after_hours_booking = allowed` treats every hour as bookable; `never` and `deferred`
    both fail this check identically — the distinction between refusing an after-hours slot
    outright and deferring it to staff is not a difference in whether autonomous booking
    happens (it never does for either value), only in messaging, which is the agent's job."""
    if not _envelope_enforced():
        return True, None
    if policy.after_hours_booking == "allowed":
        return True, None

    weekday = start_ts.weekday()  # Monday=0 ... Sunday=6
    if weekday <= 4:
        window = policy.business_hours.get("mon_fri")
    elif weekday == 5:
        window = policy.business_hours.get("sat")
    else:
        window = policy.business_hours.get("sun")

    if not window:
        return False, Reason.OUTSIDE_BUSINESS_HOURS.value

    open_t, close_t = _parse_hhmm(window["open"]), _parse_hhmm(window["close"])
    if open_t <= start_ts.time() < close_t:
        return True, None
    return False, Reason.OUTSIDE_BUSINESS_HOURS.value


def check_lead_time(policy: Policy, start_ts: datetime, now: datetime) -> tuple[bool, str | None]:
    if not _envelope_enforced():
        return True, None
    if start_ts - now >= timedelta(hours=policy.min_lead_time_hours):
        return True, None
    return False, Reason.LEAD_TIME.value


def check_booking_window(policy: Policy, start_ts: datetime, now: datetime) -> tuple[bool, str | None]:
    if not _envelope_enforced():
        return True, None
    if start_ts - now <= timedelta(days=policy.max_booking_window_days):
        return True, None
    return False, Reason.BOOKING_WINDOW.value


def check_balance_hold(policy: Policy, balance_cents: int) -> tuple[bool, str | None]:
    if not _envelope_enforced():
        return True, None
    if balance_cents <= policy.blocking_balance_above:
        return True, None
    return False, Reason.BALANCE_HOLD.value


def deposit_required(policy: Policy, price_cents: int | None) -> bool:
    return price_cents is not None and price_cents > policy.deposit_required_above


def check_discount(policy: Policy, discount_pct: int) -> tuple[bool, str | None]:
    if discount_pct <= policy.max_discount_pct:
        return True, None
    return False, Reason.DISCOUNT_CAP.value


def cancellation_fee_applies(policy: Policy, appointment_start_ts: datetime, now: datetime) -> bool:
    return appointment_start_ts - now < timedelta(hours=policy.cancellation_fee_window_hrs)


def first_envelope_failure(checks: list[tuple[bool, str | None]]) -> str | None:
    """`checks` is an ordered list of (ok, reason_code) pairs -- one per envelope condition a
    booking/reschedule must satisfy. Returns the reason code of the first one that failed, or
    None if they all passed. Order matters: it's the order failures get reported in, so put the
    most specific/actionable check first. Shared by book_appointment, reschedule_appointment,
    and book_appointment_for_customer, which each assembled this same
    build-a-list-then-find-the-first-failure idiom independently before this existed."""
    return next((code for ok, code in checks if not ok), None)
