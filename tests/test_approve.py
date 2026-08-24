"""approve.py: the only place a QUEUED request's underlying rows can actually change. Every
test here starts from a real pending_requests row produced by the normal tool path (via
dispatch), not a hand-built fixture, so this exercises the full queue -> approve loop."""

from __future__ import annotations

from datetime import datetime, timedelta

import approve
from db.database import get_session
from db.models import Appointment, Customer, Invoice, PendingRequest
from tools.dispatcher import Decision, dispatch
from tools.principal import Principal
from tools.registry_c import REGISTRY_C
from tools.registry_s import REGISTRY_S

DISPATCHER = "dispatcher"
MANAGER = "manager"


def _after_hours_start():
    d = datetime.now() + timedelta(days=5)
    while d.weekday() > 4:
        d += timedelta(days=1)
    return d.replace(hour=20, minute=0, second=0, microsecond=0)


def test_approve_queued_booking_executes_and_marks_the_request(edge_db_with_policy):
    queued = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=2, start_ts=_after_hours_start(),
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    assert queued["decision"] == Decision.QUEUED.value

    result = approve.approve(queued["request_id"], approver_role=DISPATCHER, approver_id=1)
    assert result["status"] == "executed"
    assert result["result"]["decision"] == Decision.EXECUTED.value
    with get_session() as session:
        appt = session.get(Appointment, result["result"]["appointment_id"])
        assert appt.customer_id == 14
        assert appt.status == "scheduled"
        assert session.get(PendingRequest, queued["request_id"]).status == "executed"


def test_approve_fall_forward_booking_uses_the_customer_created_at_queue_time(edge_db_with_policy):
    queued = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=None),
        service_item_id=2, start_ts=_after_hours_start(),
        name="Evasive Caller", email="evasive2@example.com", phone="619-555-7777", address="9 Nowhere Ave",
    )
    assert queued["decision"] == Decision.QUEUED.value
    customer_id = queued["customer_id"]

    result = approve.approve(queued["request_id"], approver_role=DISPATCHER, approver_id=1)
    assert result["status"] == "executed"
    with get_session() as session:
        appt = session.get(Appointment, result["result"]["appointment_id"])
        assert appt.customer_id == customer_id
        assert session.get(Customer, customer_id) is not None


def test_reject_leaves_everything_untouched(edge_db_with_policy):
    queued = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=2, start_ts=_after_hours_start(),
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    result = approve.reject(queued["request_id"], approver_role=DISPATCHER, approver_id=1)
    assert result["status"] == "rejected"
    with get_session() as session:
        assert session.query(Appointment).filter(Appointment.customer_id == 14).count() == 1  # only the seeded one
        assert session.get(PendingRequest, queued["request_id"]).status == "rejected"


def test_manager_gated_tool_refuses_a_dispatcher_approver(edge_db_with_policy):
    queued = dispatch(REGISTRY_S, "write_off_balance", Principal(type="staff", id=1, role="dispatcher"), customer_id=13, note="test")
    result = approve.approve(queued["request_id"], approver_role=DISPATCHER, approver_id=1)
    assert "error" in result
    with get_session() as session:
        assert session.get(PendingRequest, queued["request_id"]).status == "pending"
        assert session.get(Customer, 13).balance_cents == 32000  # untouched


def test_manager_approves_write_off_balance(edge_db_with_policy):
    queued = dispatch(REGISTRY_S, "write_off_balance", Principal(type="staff", id=1, role="dispatcher"), customer_id=13, note="test")
    result = approve.approve(queued["request_id"], approver_role=MANAGER, approver_id=2)
    assert result["status"] == "executed"
    with get_session() as session:
        assert session.get(Customer, 13).balance_cents == 0
        assert session.get(Invoice, 1).status == "void"


def test_manager_approves_void_invoice(edge_db_with_policy):
    queued = dispatch(REGISTRY_S, "void_invoice", Principal(type="staff", id=1, role="dispatcher"), invoice_id=2)
    result = approve.approve(queued["request_id"], approver_role=MANAGER, approver_id=2)
    assert result["status"] == "executed"
    with get_session() as session:
        assert session.get(Invoice, 2).status == "void"
        assert session.get(Customer, 14).balance_cents == 0


def test_manager_approves_merge_customers(edge_db_with_policy):
    queued = dispatch(REGISTRY_S, "merge_customers", Principal(type="staff", id=1, role="dispatcher"), survivor_id=1, loser_id=2)
    result = approve.approve(queued["request_id"], approver_role=MANAGER, approver_id=2)
    assert result["status"] == "executed"
    with get_session() as session:
        assert session.get(Customer, 2).merged_into_id == 1


def test_manager_approves_over_cap_discount(edge_db_with_policy):
    created = dispatch(REGISTRY_S, "create_invoice", Principal(type="staff", id=1, role="dispatcher"), customer_id=1, line_items=[{"unit_price_cents": 10000}])
    queued = dispatch(REGISTRY_S, "apply_discount", Principal(type="staff", id=1, role="dispatcher"), invoice_id=created["invoice_id"], discount_pct=50)
    assert queued["decision"] == Decision.QUEUED.value

    result = approve.approve(queued["request_id"], approver_role=MANAGER, approver_id=2)
    assert result["status"] == "executed"
    with get_session() as session:
        assert session.get(Invoice, created["invoice_id"]).total_cents == 5000


def test_dispatcher_can_approve_a_callback_request(edge_db_with_policy):
    """request_human_callback's approver is 'staff' generically -- no manager gate."""
    queued = dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="Call me back.")
    result = approve.approve(queued["request_id"], approver_role=DISPATCHER, approver_id=1)
    assert result["status"] == "executed"


def test_approving_an_already_resolved_request_errors(edge_db_with_policy):
    queued = dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="x")
    approve.approve(queued["request_id"], approver_role=DISPATCHER, approver_id=1)
    second = approve.approve(queued["request_id"], approver_role=DISPATCHER, approver_id=1)
    assert "error" in second


def test_approving_unknown_request_id_errors(edge_db_with_policy):
    result = approve.approve(999999, approver_role=MANAGER, approver_id=1)
    assert "error" in result


def test_cli_main_approve_and_reject(edge_db_with_policy, capsys):
    queued = dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="x")
    exit_code = approve.main([str(queued["request_id"]), "--approve", "--role", "dispatcher"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"status": "executed"' in out

    queued2 = dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="y")
    exit_code = approve.main([str(queued2["request_id"]), "--reject", "--role", "dispatcher"])
    assert exit_code == 0
    with get_session() as session:
        assert session.get(PendingRequest, queued2["request_id"]).status == "rejected"


def test_approve_writes_exactly_one_audit_row(edge_db_with_policy):
    from db.models import AuditLog

    queued = dispatch(REGISTRY_S, "void_invoice", Principal(type="staff", id=1, role="dispatcher"), invoice_id=2)
    approve.approve(queued["request_id"], approver_role=MANAGER, approver_id=2)
    with get_session() as session:
        rows = session.query(AuditLog).filter(AuditLog.tool == "void_invoice").all()
        # one QUEUED row from the original call, one EXECUTED row from approval
        assert len(rows) == 2
        assert {r.decision for r in rows} == {Decision.QUEUED.value, Decision.EXECUTED.value}
