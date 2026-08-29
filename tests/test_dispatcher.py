"""dispatch() in isolation, using a small fake registry — real registry/tool behavior is
covered in test_registry_c.py / test_registry_s.py. What matters here is purely the boundary:
membership, role gating, and pass-through, each with the correct audit trail."""

from __future__ import annotations

from datetime import datetime

from db.database import get_session
from db.models import AuditLog
from tools.dispatcher import Decision, ToolSpec, dispatch
from tools.principal import Principal
from tools.reasons import Reason


def _echo_tool(*, principal, run_id=None, **kwargs):
    return {"decision": Decision.EXECUTED.value, "principal_id": principal.id, "kwargs": kwargs}


def _echo_with_datetime(*, principal, run_id=None, start_ts: datetime, label: str = ""):
    return {"decision": Decision.EXECUTED.value, "start_ts": start_ts, "label": label}


FAKE_REGISTRY = {
    "echo": ToolSpec(fn=_echo_tool, tier=0),
    "manager_only": ToolSpec(fn=_echo_tool, tier=2, min_role="manager"),
    "echo_datetime": ToolSpec(fn=_echo_with_datetime, tier=0),
}


def test_dispatch_calls_the_tool_when_everything_checks_out(db_path_in_memory):
    principal = Principal(type="customer", id=1)
    result = dispatch(FAKE_REGISTRY, "echo", principal, foo="bar")
    assert result["decision"] == Decision.EXECUTED.value
    assert result["principal_id"] == 1
    assert result["kwargs"] == {"foo": "bar"}


def test_dispatch_denies_and_logs_tool_not_in_registry(db_path):
    principal = Principal(type="customer", id=1)
    result = dispatch(FAKE_REGISTRY, "merge_customers", principal, run_id="run-x")
    assert result == {"decision": Decision.DENIED.value, "reason": Reason.NOT_IN_REGISTRY.value, "tool": "merge_customers"}

    with get_session() as session:
        row = session.query(AuditLog).one()
        assert row.tool == "merge_customers"
        assert row.decision == Decision.DENIED.value
        assert row.reason == Reason.NOT_IN_REGISTRY.value
        assert row.run_id == "run-x"
        assert row.principal_type == "customer"
        assert row.principal_id == 1


def test_dispatch_never_calls_the_tool_function_for_an_absent_tool(db_path, monkeypatch):
    calls = []
    monkeypatch.setitem(FAKE_REGISTRY, "echo", ToolSpec(fn=lambda **kw: calls.append(kw), tier=0))
    dispatch(FAKE_REGISTRY, "not_a_real_tool", Principal(type="customer", id=1))
    assert calls == []


def test_dispatch_denies_insufficient_role(db_path):
    dispatcher_role = Principal(type="staff", id=1, role="dispatcher")
    result = dispatch(FAKE_REGISTRY, "manager_only", dispatcher_role)
    assert result["decision"] == Decision.DENIED.value
    assert result["reason"] == Reason.INSUFFICIENT_ROLE.value

    with get_session() as session:
        row = session.query(AuditLog).one()
        assert row.reason == Reason.INSUFFICIENT_ROLE.value
        assert row.declared_tier == 2


def test_dispatch_allows_sufficient_role(db_path_in_memory):
    manager = Principal(type="staff", id=2, role="manager")
    result = dispatch(FAKE_REGISTRY, "manager_only", manager)
    assert result["decision"] == Decision.EXECUTED.value

    owner = Principal(type="staff", id=3, role="owner")
    result = dispatch(FAKE_REGISTRY, "manager_only", owner)
    assert result["decision"] == Decision.EXECUTED.value


def test_dispatch_role_gate_never_applies_to_a_customer_principal(db_path):
    """A tool with min_role set is staff-only by construction (customers never have a role),
    so a customer principal always fails the gate rather than bypassing it."""
    customer = Principal(type="customer", id=1)
    result = dispatch(FAKE_REGISTRY, "manager_only", customer)
    assert result["decision"] == Decision.DENIED.value
    assert result["reason"] == Reason.INSUFFICIENT_ROLE.value


def test_dispatch_coerces_an_iso_string_into_a_real_datetime(db_path_in_memory):
    """A model's tool_use.input carries every datetime argument as a JSON string (agent/
    schemas.py declares them exactly that way) -- dispatch() must hand the tool function a real
    datetime, since the tool does arithmetic on it directly (start_ts + timedelta(...))."""
    result = dispatch(FAKE_REGISTRY, "echo_datetime", Principal(type="customer", id=1), start_ts="2026-09-01T10:00:00")
    assert result["start_ts"] == datetime(2026, 9, 1, 10, 0)
    assert isinstance(result["start_ts"], datetime)


def test_dispatch_leaves_a_real_datetime_argument_untouched(db_path_in_memory):
    real_dt = datetime(2026, 9, 1, 10, 0)
    result = dispatch(FAKE_REGISTRY, "echo_datetime", Principal(type="customer", id=1), start_ts=real_dt)
    assert result["start_ts"] is real_dt


def test_dispatch_does_not_coerce_a_non_datetime_string_argument(db_path_in_memory):
    """Only parameters the tool function itself types as `datetime` get coerced -- an ordinary
    string argument is passed through exactly as given."""
    result = dispatch(
        FAKE_REGISTRY, "echo_datetime", Principal(type="customer", id=1),
        start_ts="2026-09-01T10:00:00", label="2026-09-01T10:00:00",
    )
    assert result["label"] == "2026-09-01T10:00:00"  # still a string, not parsed


def test_dispatch_normalizes_a_tz_aware_iso_string_to_naive_utc(db_path_in_memory):
    """Claude reliably includes a "Z" or explicit offset in ISO-8601 strings it emits -- naively
    coercing that produces a tz-aware datetime, which raises TypeError the moment a tool compares
    it against db.seed_common.now_utc() (naive by convention). Reproduces the always-on
    get_availability crash found in suite-1787905540: every one of its 4 affected cases hit this
    exact "can't compare offset-naive and offset-aware datetimes" error on every retry."""
    result = dispatch(
        FAKE_REGISTRY, "echo_datetime", Principal(type="customer", id=1),
        start_ts="2026-09-01T10:00:00Z",
    )
    assert result["start_ts"] == datetime(2026, 9, 1, 10, 0)
    assert result["start_ts"].tzinfo is None


def test_dispatch_keeps_the_wall_clock_hour_when_the_model_qualifies_it_with_an_offset(db_path_in_memory):
    """Every timestamp in this system is naive local business time, so an offset the model
    attached is noise to discard, not a frame to convert from. 10:00 stays 10:00.

    This previously converted to UTC, which moved the hour: a run of the eval suite carried
    `Z`, `-00:00`, `-04:00` and `-07:00` across different cases, and the non-zero ones shifted
    every booking and availability window by four or seven hours."""
    result = dispatch(
        FAKE_REGISTRY, "echo_datetime", Principal(type="customer", id=1),
        start_ts="2026-09-01T10:00:00-07:00",
    )
    assert result["start_ts"] == datetime(2026, 9, 1, 10, 0)
    assert result["start_ts"].tzinfo is None


def test_dispatch_coerces_every_offset_the_model_uses_to_the_same_naive_value(db_path_in_memory):
    """The specific defect was inconsistency: the same requested hour landed on a different
    stored value depending on which timezone the model happened to guess that turn."""
    naive = datetime(2026, 9, 1, 10, 0)
    for qualifier in ("", "Z", "+00:00", "-00:00", "-04:00", "-07:00", "+05:30"):
        result = dispatch(
            FAKE_REGISTRY, "echo_datetime", Principal(type="customer", id=1),
            start_ts=f"2026-09-01T10:00:00{qualifier}",
        )
        assert result["start_ts"] == naive, f"offset {qualifier!r} changed the wall-clock hour"
        assert result["start_ts"].tzinfo is None
