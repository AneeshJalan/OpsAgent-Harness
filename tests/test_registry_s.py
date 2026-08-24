"""Registry S tool behavior — the staff-facing surface. Less about escalation matrices (staff
is trusted) and more about: fail-closed scope on every read, the role gate on record_payment,
and that Tier-3 tools genuinely never touch state on their own."""

from __future__ import annotations

from datetime import datetime, timedelta

from db.database import get_session
from db.models import AuditLog, Customer, Invoice, PendingRequest
from tools.dispatcher import Decision, dispatch
from tools.principal import Principal
from tools.registry_s import (
    REGISTRY_S,
    _execute_apply_discount,
    _execute_merge_customers,
    _execute_void_invoice,
    _execute_write_off_balance,
)

DISPATCHER = Principal(type="staff", id=1, role="dispatcher")
MANAGER = Principal(type="staff", id=2, role="manager")


def test_registry_s_has_exactly_eighteen_tools_with_correct_tiers(edge_db_with_policy):
    assert len(REGISTRY_S) == 18
    expected_tiers = {
        "search_customers": 0, "get_customer_detail": 0, "list_appointments": 0,
        "get_schedule": 0, "list_invoices": 0, "find_duplicate_candidates": 0,
        "find_schedule_conflicts": 0, "book_appointment_for_customer": 1,
        "reassign_technician": 1, "add_internal_note": 1, "create_invoice": 2,
        "send_invoice": 2, "apply_discount": 2, "cancel_appointment_with_notice": 2,
        "record_payment": 2, "write_off_balance": 3, "void_invoice": 3, "merge_customers": 3,
    }
    assert {name: spec.tier for name, spec in REGISTRY_S.items()} == expected_tiers
    assert REGISTRY_S["record_payment"].min_role == "manager"
    assert all(spec.min_role is None for name, spec in REGISTRY_S.items() if name != "record_payment")


def test_search_customers_requires_a_query(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "search_customers", DISPATCHER, query="")
    assert result["decision"] == "denied"
    assert result["reason"] == "invalid_argument"


def test_search_customers_finds_by_partial_name(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "search_customers", DISPATCHER, query="jonathan reyes")
    ids = {c["id"] for c in result["customers"]}
    assert {1, 2}.issubset(ids)
    # trusted registry: full contact fields ARE present, unlike anything Registry C exposes.
    assert result["customers"][0]["phone"] is not None


def test_list_appointments_requires_a_bounded_range(edge_db_with_policy):
    result = dispatch(
        REGISTRY_S, "list_appointments", DISPATCHER,
        range_start=datetime(2026, 8, 1), range_end=datetime(2026, 9, 1), technician_id=7,
    )
    ids = {a["id"] for a in result["appointments"]}
    assert ids == {1, 2}  # technician 7's planted double-booking


def test_list_invoices_rejects_unscoped_call(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "list_invoices", DISPATCHER)
    assert result["decision"] == "denied"
    assert result["reason"] == "invalid_argument"


def test_list_invoices_scoped_by_customer(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "list_invoices", DISPATCHER, customer_id=13)
    assert {i["id"] for i in result["invoices"]} == {1}


def test_find_duplicate_candidates_surfaces_planted_pairs(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "find_duplicate_candidates", DISPATCHER)
    by_pair = {frozenset({c["customer_id_a"], c["customer_id_b"]}): c for c in result["candidates"]}
    assert frozenset({1, 2}) in by_pair  # formatting-only
    assert frozenset({3, 4}) in by_pair  # typo
    assert frozenset({7, 8}) in by_pair  # shared phone

    # The same-address hard negative shares no exact-match signal (different unit) and must
    # not appear at all.
    assert frozenset({9, 10}) not in by_pair

    # The same-full-name hard negative IS a legitimate candidate for a human to review — this
    # tool surfaces candidates, not verdicts — but it must carry only the weak signal that's
    # actually true (name), never a phone or address match that isn't there.
    same_name = by_pair[frozenset({11, 12})]
    assert same_name["signals"] == ["similar_name:1.00"]


def test_find_schedule_conflicts_finds_the_planted_double_booking(edge_db_with_policy):
    result = dispatch(
        REGISTRY_S, "find_schedule_conflicts", DISPATCHER,
        range_start=datetime(2026, 8, 1), range_end=datetime(2026, 9, 1),
    )
    found = {(c["appointment_id_a"], c["appointment_id_b"]) for c in result["conflicts"]}
    assert (1, 2) in found or (2, 1) in found


def test_book_appointment_for_customer_denied_without_override(edge_db_with_policy):
    after_hours = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0) + timedelta(days=5)
    result = dispatch(
        REGISTRY_S, "book_appointment_for_customer", DISPATCHER,
        customer_id=14, service_item_id=2, start_ts=after_hours,
    )
    assert result["decision"] == "denied"
    assert result["reason"] == "outside_business_hours"


def test_book_appointment_for_customer_executes_with_override(edge_db_with_policy):
    after_hours = datetime.now().replace(hour=20, minute=0, second=0, microsecond=0) + timedelta(days=5)
    result = dispatch(
        REGISTRY_S, "book_appointment_for_customer", DISPATCHER,
        customer_id=14, service_item_id=2, start_ts=after_hours, override_business_hours=True,
    )
    assert result["decision"] == "executed"


def test_reassign_technician_reports_skill_mismatch_but_still_executes(edge_db_with_policy):
    """Appointment 1 requires hvac (service_item 4); technician 6 only has drain_cleaning."""
    result = dispatch(REGISTRY_S, "reassign_technician", DISPATCHER, appointment_id=1, technician_id=6)
    assert result["decision"] == "executed"
    assert "technician lacks the skill this service requires" in result["warnings"]
    with get_session() as session:
        from db.models import Appointment
        assert session.get(Appointment, 1).technician_id == 6


def test_add_internal_note_appends_with_timestamp(edge_db_with_policy):
    dispatch(REGISTRY_S, "add_internal_note", DISPATCHER, customer_id=1, note_text="Called about billing.")
    with get_session() as session:
        note = session.get(Customer, 1).internal_notes
    assert "Called about billing." in note
    assert "staff:1" in note


def test_create_invoice_send_invoice_and_record_payment_happy_path(edge_db_with_policy):
    created = dispatch(
        REGISTRY_S, "create_invoice", DISPATCHER, customer_id=1,
        line_items=[{"service_item_id": 1, "description": "Drain Cleaning", "qty": 1, "unit_price_cents": 15000}],
    )
    assert created["decision"] == "executed"
    invoice_id = created["invoice_id"]
    assert created["total_cents"] == 15000

    sent = dispatch(REGISTRY_S, "send_invoice", DISPATCHER, invoice_id=invoice_id)
    assert sent["decision"] == "executed"

    # dispatcher role cannot record payment
    denied = dispatch(REGISTRY_S, "record_payment", DISPATCHER, invoice_id=invoice_id, processor_ref="ch_1", amount_cents=15000)
    assert denied["decision"] == "denied"
    assert denied["reason"] == "insufficient_role"

    paid = dispatch(REGISTRY_S, "record_payment", MANAGER, invoice_id=invoice_id, processor_ref="ch_1", amount_cents=15000)
    assert paid["decision"] == "executed"
    with get_session() as session:
        invoice = session.get(Invoice, invoice_id)
        assert invoice.status == "paid"
        assert invoice.processor_ref == "ch_1"


def test_record_payment_rejects_partial_amount(edge_db_with_policy):
    created = dispatch(
        REGISTRY_S, "create_invoice", DISPATCHER, customer_id=1,
        line_items=[{"unit_price_cents": 10000}],
    )
    dispatch(REGISTRY_S, "send_invoice", DISPATCHER, invoice_id=created["invoice_id"])
    result = dispatch(REGISTRY_S, "record_payment", MANAGER, invoice_id=created["invoice_id"], processor_ref="x", amount_cents=9999)
    assert result["decision"] == "denied"
    assert result["reason"] == "invalid_argument"


def test_apply_discount_executes_within_cap_and_queues_above_it(edge_db_with_policy):
    created = dispatch(
        REGISTRY_S, "create_invoice", DISPATCHER, customer_id=1, line_items=[{"unit_price_cents": 10000}],
    )
    invoice_id = created["invoice_id"]

    within_cap = dispatch(REGISTRY_S, "apply_discount", DISPATCHER, invoice_id=invoice_id, discount_pct=10)
    assert within_cap["decision"] == "executed"
    assert within_cap["new_total_cents"] == 9000

    above_cap = dispatch(REGISTRY_S, "apply_discount", DISPATCHER, invoice_id=invoice_id, discount_pct=50)
    assert above_cap["decision"] == "queued"
    assert above_cap["reason"] == "discount_cap"
    with get_session() as session:
        # invoice untouched by the queued (not yet approved) request
        assert session.get(Invoice, invoice_id).total_cents == 9000


def test_write_off_balance_queues_and_does_not_touch_state(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "write_off_balance", DISPATCHER, customer_id=13, note="Bad debt.")
    assert result["decision"] == "queued"
    with get_session() as session:
        assert session.get(Customer, 13).balance_cents == 32000  # unchanged
        assert session.get(PendingRequest, result["request_id"]).status == "pending"


def test_write_off_balance_executor_voids_unpaid_invoices(edge_db_with_policy):
    with get_session() as session:
        result = _execute_write_off_balance(session, 13)
        session.commit()
    assert result["invoices_voided"] == [1]
    with get_session() as session:
        assert session.get(Invoice, 1).status == "void"
        assert session.get(Customer, 13).balance_cents == 0


def test_void_invoice_queues_then_executor_applies_it(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "void_invoice", DISPATCHER, invoice_id=2)
    assert result["decision"] == "queued"
    with get_session() as session:
        assert session.get(Invoice, 2).status == "sent"  # unchanged until approved

    with get_session() as session:
        _execute_void_invoice(session, 2)
        session.commit()
    with get_session() as session:
        assert session.get(Invoice, 2).status == "void"
        assert session.get(Customer, 14).balance_cents == 0


def test_merge_customers_queues_with_a_field_diff_then_executor_applies_it(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "merge_customers", DISPATCHER, survivor_id=1, loser_id=2)
    assert result["decision"] == "queued"
    assert "customer:2" in result["preview_text"] or "2" in result["preview_text"]
    with get_session() as session:
        assert session.get(Customer, 2).merged_into_id is None  # not yet merged

    with get_session() as session:
        _execute_merge_customers(session, survivor_id=1, loser_id=2)
        session.commit()
    with get_session() as session:
        assert session.get(Customer, 2).merged_into_id == 1


def test_merge_customers_rejects_self_merge(edge_db_with_policy):
    result = dispatch(REGISTRY_S, "merge_customers", DISPATCHER, survivor_id=1, loser_id=1)
    assert result["decision"] == "denied"
    assert result["reason"] == "invalid_argument"


def test_tier_3_tools_never_write_state_outside_the_executor(edge_db_with_policy):
    """Calling the public tools for every Tier-3 action must leave the target rows untouched
    -- only the paired _execute_* helper (used by approve.py after approval) does."""
    dispatch(REGISTRY_S, "void_invoice", DISPATCHER, invoice_id=3)
    dispatch(REGISTRY_S, "write_off_balance", DISPATCHER, customer_id=9, note="test")
    dispatch(REGISTRY_S, "merge_customers", DISPATCHER, survivor_id=11, loser_id=12)
    with get_session() as session:
        assert session.get(Invoice, 3).status == "paid"  # seeded as paid, untouched
        assert session.get(Customer, 9).balance_cents == 0  # already 0, untouched
        assert session.get(Customer, 12).merged_into_id is None


def test_staff_tool_call_writes_exactly_one_audit_row(edge_db_with_policy):
    dispatch(REGISTRY_S, "add_internal_note", DISPATCHER, customer_id=1, note_text="x")
    with get_session() as session:
        assert session.query(AuditLog).filter(AuditLog.tool == "add_internal_note").count() == 1
