"""evals/checks/state.py: snapshot/diff/check_state. Exercised against a real DB and real tool
calls (not hand-built rows) wherever practical, so a passing test means the checker actually
sees what a real run produces.
"""

from __future__ import annotations

from db.database import get_session
from db.models import Customer
from evals.checks.state import check_state, diff_snapshots, rows_as_dicts, snapshot
from tools.dispatcher import dispatch
from tools.principal import Principal
from tools.registry_c import REGISTRY_C


def test_snapshot_captures_every_table_with_matching_row_counts(edge_db):
    snap = snapshot(edge_db)
    with get_session() as session:
        expected = session.query(Customer).count()
    assert "customers" in snap
    assert len(snap["customers"]["rows"]) == expected
    assert "audit_log" in snap  # present even when empty
    assert "columns" in snap["customers"] and "id" in snap["customers"]["columns"]


def test_rows_as_dicts_reconstructs_named_fields(edge_db):
    snap = snapshot(edge_db)
    dicts = rows_as_dicts(snap["customers"])
    assert all("name" in d and "balance_cents" in d for d in dicts)
    with get_session() as session:
        expected_ids = {c.id for c in session.query(Customer).all()}
    assert {d["id"] for d in dicts} == expected_ids


def test_diff_is_empty_when_nothing_changed(edge_db):
    before = snapshot(edge_db)
    after = snapshot(edge_db)
    assert diff_snapshots(before, after) == {}


def test_diff_detects_a_real_tool_call_side_effect(edge_db):
    """request_human_callback always queues -- one new pending_requests row, one new audit_log
    row, nothing else."""
    before = snapshot(edge_db)
    dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="call me")
    after = snapshot(edge_db)

    diff = diff_snapshots(before, after)
    assert set(diff) == {"pending_requests", "audit_log"}
    assert len(diff["pending_requests"]["added"]) == 1
    assert diff["pending_requests"]["removed"] == []
    assert len(diff["audit_log"]["added"]) == 1


def test_diff_represents_a_modified_row_as_one_removed_and_one_added(edge_db):
    before = snapshot(edge_db)
    with get_session() as session:
        customer = session.query(Customer).first()
        customer.name = "Renamed Customer"
        session.commit()
    after = snapshot(edge_db)

    diff = diff_snapshots(before, after)
    assert len(diff["customers"]["added"]) == 1
    assert len(diff["customers"]["removed"]) == 1


def test_check_state_unchanged_passes_when_nothing_changed(edge_db):
    before = snapshot(edge_db)
    after = snapshot(edge_db)
    result = check_state(before, after, {"unchanged": True})
    assert result.passed is True


def test_check_state_unchanged_passes_on_a_denied_call_despite_the_denial_writing_an_audit_row(edge_db):
    """The motivating case for the audit_log exemption: a customer tries to read someone else's
    appointments, gets denied, and no business state changes -- but the denial itself still (and
    correctly) writes an audit_log row. guards.state: {unchanged: true} must still pass; a
    checker that didn't exempt audit_log would fail every denial case in the suite."""
    before = snapshot(edge_db)
    result_dict = dispatch(REGISTRY_C, "get_my_appointments", Principal(type="customer", id=1), customer_id=999)
    assert result_dict["decision"] == "denied"  # sanity: the call really was denied
    after = snapshot(edge_db)

    diff = diff_snapshots(before, after)
    assert "audit_log" in diff  # the denial really did write an audit row

    result = check_state(before, after, {"unchanged": True})
    assert result.passed is True, result.detail


def test_check_state_unchanged_fails_when_something_changed(edge_db):
    before = snapshot(edge_db)
    dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="x")
    after = snapshot(edge_db)
    result = check_state(before, after, {"unchanged": True})
    assert result.passed is False
    assert "pending_requests" in result.detail or "audit_log" in result.detail


def test_check_state_explicit_change_set_passes_when_it_matches(edge_db):
    before = snapshot(edge_db)
    dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="x")
    after = snapshot(edge_db)
    result = check_state(before, after, {
        "tables": {"pending_requests": {"added": 1}, "audit_log": {"added": 1}}
    })
    assert result.passed is True, result.detail


def test_check_state_explicit_change_set_fails_on_wrong_count(edge_db):
    before = snapshot(edge_db)
    dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="x")
    after = snapshot(edge_db)
    result = check_state(before, after, {
        "tables": {"pending_requests": {"added": 2}, "audit_log": {"added": 1}}
    })
    assert result.passed is False


def test_check_state_fails_on_a_business_table_the_case_did_not_expect_to_change(edge_db_with_policy):
    """A queue-then-approve sequence touches pending_requests, invoices, AND customers -- a
    case that only names pending_requests is missing real business-table changes, which must
    fail (unlike the audit_log exemption, these are exactly the changes this checker exists to
    catch)."""
    import approve
    from tools.dispatcher import dispatch as _dispatch
    from tools.principal import Principal as _Principal
    from tools.registry_s import REGISTRY_S

    before = snapshot(edge_db_with_policy)
    queued = _dispatch(REGISTRY_S, "write_off_balance", _Principal(type="staff", id=1, role="dispatcher"), customer_id=13, note="x")
    approve.approve(queued["request_id"], approver_role="manager", approver_id=2)
    after = snapshot(edge_db_with_policy)

    result = check_state(before, after, {"tables": {"pending_requests": {"added": 1, "removed": 1}}})
    assert result.passed is False
    assert "invoices" in result.detail or "customers" in result.detail
    assert "audit_log" not in result.detail  # audit_log is never itself the reported culprit
