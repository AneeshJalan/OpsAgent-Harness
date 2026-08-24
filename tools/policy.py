"""The business's operating envelope, read from policy_config, plus the individual checks
that decide whether a request stays inside it. Each check returns (ok, reason_code | None) —
turning a failed check into QUEUED, NEEDS_CONFIRM, or DENIED is the calling tool's job, not
this module's. Keeping that split means the same check can back different runtime decisions
in different tools (e.g. a failed business-hours check queues a customer booking but is simply
overridable by staff via book_appointment_for_customer).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from sqlalchemy.orm import Session

from db.models import PolicyConfig
from tools.reasons import Reason


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
    if start_ts - now >= timedelta(hours=policy.min_lead_time_hours):
        return True, None
    return False, Reason.LEAD_TIME.value


def check_booking_window(policy: Policy, start_ts: datetime, now: datetime) -> tuple[bool, str | None]:
    if start_ts - now <= timedelta(days=policy.max_booking_window_days):
        return True, None
    return False, Reason.BOOKING_WINDOW.value


def check_balance_hold(policy: Policy, balance_cents: int) -> tuple[bool, str | None]:
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
