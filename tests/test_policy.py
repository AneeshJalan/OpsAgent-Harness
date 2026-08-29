"""Policy envelope checks, in isolation from any tool. These are the primitives book_appointment
and friends compose — get them right here so a booking-flow test failure means the booking
logic is wrong, not that lead time was computed incorrectly three call-frames down."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from db.database import get_session
from tools.policy import (
    cancellation_fee_applies,
    check_balance_hold,
    check_booking_window,
    check_business_hours,
    check_discount,
    check_lead_time,
    deposit_required,
    first_envelope_failure,
    load_policy,
)
from tools.reasons import Reason

_ENV_VAR = "POLICY_ENFORCEMENT"


def _policy(policy_only_db):
    with get_session() as session:
        return load_policy(session)


def test_load_policy_matches_seeded_defaults(policy_only_db):
    p = _policy(policy_only_db)
    assert p.min_lead_time_hours == 4
    assert p.max_booking_window_days == 60
    assert p.auto_book_enabled is True
    assert p.deposit_required_above == 50000
    assert p.blocking_balance_above == 25000
    assert p.max_discount_pct == 15
    assert p.cancellation_fee_window_hrs == 24
    assert p.after_hours_booking == "deferred"
    assert p.identity_mode == "weak"
    assert p.business_hours["mon_fri"] == {"open": "08:00", "close": "18:00"}
    assert p.business_hours["sun"] is None


def test_business_hours_weekday_inside_and_outside(policy_only_db):
    p = _policy(policy_only_db)
    # Tuesday 2026-08-25, 10:00 -> inside Mon-Fri 08:00-18:00
    ok, reason = check_business_hours(p, datetime(2026, 8, 25, 10, 0))
    assert ok and reason is None
    # Same Tuesday, 19:00 -> after close
    ok, reason = check_business_hours(p, datetime(2026, 8, 25, 19, 0))
    assert not ok and reason == Reason.OUTSIDE_BUSINESS_HOURS.value


def test_business_hours_sunday_closed(policy_only_db):
    p = _policy(policy_only_db)
    # 2026-08-23 is a Sunday; sun window is null in the seeded defaults.
    ok, reason = check_business_hours(p, datetime(2026, 8, 23, 12, 0))
    assert not ok and reason == Reason.OUTSIDE_BUSINESS_HOURS.value


def test_business_hours_saturday_shorter_window(policy_only_db):
    p = _policy(policy_only_db)
    # 2026-08-29 is a Saturday; Sat window is 09:00-14:00.
    ok, _ = check_business_hours(p, datetime(2026, 8, 29, 13, 30))
    assert ok
    ok, reason = check_business_hours(p, datetime(2026, 8, 29, 15, 0))
    assert not ok and reason == Reason.OUTSIDE_BUSINESS_HOURS.value


def test_after_hours_allowed_bypasses_the_window(policy_only_db):
    p = _policy(policy_only_db)
    p = p.__class__(**{**p.__dict__, "after_hours_booking": "allowed"})
    ok, reason = check_business_hours(p, datetime(2026, 8, 23, 3, 0))  # Sunday 3am
    assert ok and reason is None


def test_lead_time(policy_only_db):
    p = _policy(policy_only_db)
    now = datetime(2026, 8, 25, 8, 0)
    ok, _ = check_lead_time(p, now, now)
    assert not ok  # zero lead time, needs >= 4 hours
    ok, reason = check_lead_time(p, now + timedelta(hours=5), now)
    assert ok and reason is None


def test_booking_window(policy_only_db):
    p = _policy(policy_only_db)
    now = datetime(2026, 8, 25, 8, 0)
    ok, _ = check_booking_window(p, now, now)
    assert ok
    ok, reason = check_booking_window(p, now + timedelta(days=61), now)
    assert not ok and reason == Reason.BOOKING_WINDOW.value


def test_balance_hold_threshold(policy_only_db):
    p = _policy(policy_only_db)
    ok, _ = check_balance_hold(p, 25000)  # at threshold, not above -> ok
    assert ok
    ok, reason = check_balance_hold(p, 25001)
    assert not ok and reason == Reason.BALANCE_HOLD.value


def test_deposit_required_threshold(policy_only_db):
    p = _policy(policy_only_db)
    assert deposit_required(p, 50000) is False  # at threshold, not above
    assert deposit_required(p, 50001) is True
    assert deposit_required(p, None) is False  # null price is a different failure entirely


def test_discount_cap(policy_only_db):
    p = _policy(policy_only_db)
    ok, _ = check_discount(p, 15)
    assert ok
    ok, reason = check_discount(p, 16)
    assert not ok and reason == Reason.DISCOUNT_CAP.value


def test_cancellation_fee_window(policy_only_db):
    p = _policy(policy_only_db)
    start = datetime(2026, 8, 25, 12, 0)
    assert cancellation_fee_applies(p, start, start - timedelta(hours=23)) is True
    assert cancellation_fee_applies(p, start, start - timedelta(hours=25)) is False


def test_first_envelope_failure_returns_none_when_everything_passes():
    assert first_envelope_failure([(True, "a"), (True, "b")]) is None


def test_first_envelope_failure_returns_the_first_failing_reason_in_order():
    checks = [(True, "a"), (False, "b"), (False, "c")]
    assert first_envelope_failure(checks) == "b"


# --- POLICY_ENFORCEMENT ablation switch --------------------------------------------------------


@pytest.fixture
def prompt_only_policy_enforcement():
    """Sets POLICY_ENFORCEMENT=prompt_only for one test, restoring whatever was there (including
    nothing) afterward -- mirrors evals/case_runner.py's own _ablation_policy_enforcement, but as
    a fixture rather than a context manager since every test here wants it on for its full body."""
    previous = os.environ.get(_ENV_VAR)
    os.environ[_ENV_VAR] = "prompt_only"
    yield
    if previous is None:
        os.environ.pop(_ENV_VAR, None)
    else:
        os.environ[_ENV_VAR] = previous


def test_business_hours_check_is_bypassed_under_the_ablation(policy_only_db, prompt_only_policy_enforcement):
    p = _policy(policy_only_db)
    ok, reason = check_business_hours(p, datetime(2026, 8, 23, 3, 0))  # Sunday 3am -- normally fails
    assert ok and reason is None


def test_lead_time_check_is_bypassed_under_the_ablation(policy_only_db, prompt_only_policy_enforcement):
    p = _policy(policy_only_db)
    now = datetime(2026, 8, 25, 8, 0)
    ok, reason = check_lead_time(p, now, now)  # zero lead time -- normally fails
    assert ok and reason is None


def test_booking_window_check_is_bypassed_under_the_ablation(policy_only_db, prompt_only_policy_enforcement):
    p = _policy(policy_only_db)
    now = datetime(2026, 8, 25, 8, 0)
    ok, reason = check_booking_window(p, now + timedelta(days=61), now)  # normally fails
    assert ok and reason is None


def test_balance_hold_check_is_bypassed_under_the_ablation(policy_only_db, prompt_only_policy_enforcement):
    p = _policy(policy_only_db)
    ok, reason = check_balance_hold(p, 25001)  # normally fails
    assert ok and reason is None


def test_discount_cap_is_not_part_of_the_ablation(policy_only_db, prompt_only_policy_enforcement):
    """check_discount is staff-only and was never restated in SYSTEM_C_POLICY_IN_PROMPT -- the
    ablation switch must not touch it, even when active."""
    p = _policy(policy_only_db)
    ok, reason = check_discount(p, 16)  # still fails
    assert not ok and reason == Reason.DISCOUNT_CAP.value


def test_policy_enforcement_defaults_to_fully_enforced_when_the_env_var_is_absent(policy_only_db):
    os.environ.pop(_ENV_VAR, None)
    p = _policy(policy_only_db)
    ok, reason = check_business_hours(p, datetime(2026, 8, 23, 3, 0))  # Sunday 3am
    assert not ok and reason == Reason.OUTSIDE_BUSINESS_HOURS.value


def test_policy_enforcement_ignores_any_value_other_than_prompt_only(policy_only_db):
    previous = os.environ.get(_ENV_VAR)
    os.environ[_ENV_VAR] = "code"
    try:
        p = _policy(policy_only_db)
        ok, reason = check_business_hours(p, datetime(2026, 8, 23, 3, 0))
        assert not ok and reason == Reason.OUTSIDE_BUSINESS_HOURS.value
    finally:
        if previous is None:
            os.environ.pop(_ENV_VAR, None)
        else:
            os.environ[_ENV_VAR] = previous
