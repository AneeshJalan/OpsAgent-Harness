"""System-wide invariants that must hold after every scripted sequence of tool calls, not
just at seed time. seed's own validate_seed.py checks these against a freshly-built database;
this file checks the same invariant survives a realistic run of actual tool traffic across
both registries plus the approval loop -- book, cancel, invoice, discount, pay, write off.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import approve
from db.database import get_session
from db.models import Customer, Invoice, PendingRequest
from db.seed_common import UNPAID_STATUSES
from tools.dispatcher import dispatch
from tools.principal import Principal
from tools.registry_c import REGISTRY_C
from tools.registry_s import REGISTRY_S

DISPATCHER = Principal(type="staff", id=1, role="dispatcher")
MANAGER = Principal(type="staff", id=2, role="manager")


def _weekday_at(days_ahead: int, hour: int) -> datetime:
    d = datetime.now() + timedelta(days=days_ahead)
    while d.weekday() > 4:
        d += timedelta(days=1)
    return d.replace(hour=hour, minute=0, second=0, microsecond=0)


def _assert_balance_invariant(session) -> None:
    totals: dict[int, int] = {}
    for customer_id, total_cents in session.query(Invoice.customer_id, Invoice.total_cents).filter(
        Invoice.status.in_(UNPAID_STATUSES)
    ):
        totals[customer_id] = totals.get(customer_id, 0) + total_cents
    for customer in session.query(Customer).all():
        expected = totals.get(customer.id, 0)
        assert customer.balance_cents == expected, (
            f"customer {customer.id}: balance_cents={customer.balance_cents} but "
            f"unpaid invoice total is {expected}"
        )


def test_balance_invariant_holds_after_a_mixed_sequence_of_tool_calls(edge_db_with_policy):
    with get_session() as session:
        _assert_balance_invariant(session)  # seed-time baseline

    dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=2, start_ts=_weekday_at(5, 10),
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    with get_session() as session:
        _assert_balance_invariant(session)

    created = dispatch(REGISTRY_S, "create_invoice", DISPATCHER, customer_id=9, line_items=[{"unit_price_cents": 20000}])
    with get_session() as session:
        _assert_balance_invariant(session)

    dispatch(REGISTRY_S, "apply_discount", DISPATCHER, invoice_id=created["invoice_id"], discount_pct=10)
    with get_session() as session:
        _assert_balance_invariant(session)

    dispatch(REGISTRY_S, "send_invoice", DISPATCHER, invoice_id=created["invoice_id"])
    with get_session() as session:
        _assert_balance_invariant(session)

    dispatch(REGISTRY_S, "record_payment", MANAGER, invoice_id=created["invoice_id"], processor_ref="ch_x", amount_cents=18000)
    with get_session() as session:
        _assert_balance_invariant(session)

    write_off = dispatch(REGISTRY_S, "write_off_balance", DISPATCHER, customer_id=13, note="bad debt")
    approve.approve(write_off["request_id"], approver_role="manager", approver_id=2)
    with get_session() as session:
        _assert_balance_invariant(session)

    void = dispatch(REGISTRY_S, "void_invoice", DISPATCHER, invoice_id=2)
    approve.approve(void["request_id"], approver_role="manager", approver_id=2)
    with get_session() as session:
        _assert_balance_invariant(session)


def test_no_pending_request_is_left_orphaned_after_approval_or_rejection(edge_db_with_policy):
    """Every resolved pending_requests row must have resolved_at and resolved_by set -- a row
    stuck 'pending' forever with no way to tell is its own kind of silent failure."""
    queued = dispatch(REGISTRY_S, "void_invoice", DISPATCHER, invoice_id=3)
    approve.approve(queued["request_id"], approver_role="manager", approver_id=2)
    rejected = dispatch(REGISTRY_C, "request_human_callback", Principal(type="customer", id=1), message="x")
    approve.reject(rejected["request_id"], approver_role="dispatcher", approver_id=1)

    with get_session() as session:
        for req_id in (queued["request_id"], rejected["request_id"]):
            row = session.get(PendingRequest, req_id)
            assert row.status in ("executed", "rejected")
            assert row.resolved_at is not None
            assert row.resolved_by is not None


def test_no_audit_row_exists_without_a_declared_tier_and_decision(edge_db_with_policy):
    dispatch(REGISTRY_C, "list_services", Principal(type="customer", id=None))
    dispatch(REGISTRY_S, "search_customers", DISPATCHER, query="pham")
    with get_session() as session:
        from db.models import AuditLog
        for row in session.query(AuditLog).all():
            assert row.declared_tier in (0, 1, 2, 3)
            assert row.decision in ("executed", "needs_confirm", "queued", "denied")
            assert row.outcome in ("ok", "error")
