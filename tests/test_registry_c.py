"""Registry C tool behavior — the customer-facing surface. Focused on the decision matrix
(EXECUTED / NEEDS_CONFIRM / QUEUED / DENIED) each tool can produce, and on the ownership /
identity gates that are this project's actual security boundary.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from db.database import get_session
from db.models import Appointment, Customer, PendingRequest, ServiceItem
from tools.dispatcher import Decision, dispatch
from tools.principal import Principal
from tools.reasons import Reason
from tools.registry_c import REGISTRY_C, book_appointment, cancel_appointment


def _next_weekday_at(base: datetime, hour: int) -> datetime:
    d = base
    while d.weekday() > 4:  # skip Saturday/Sunday
        d += timedelta(days=1)
    return datetime(d.year, d.month, d.day, hour, 0)


@pytest.fixture
def in_envelope_start():
    """A start_ts guaranteed to be a weekday during business hours, comfortably past the
    lead-time minimum and inside the booking window, regardless of what day tests run on."""
    return _next_weekday_at(datetime.now() + timedelta(days=5), 10)


def test_registry_c_has_exactly_ten_tools_with_correct_tiers(edge_db):
    assert len(REGISTRY_C) == 10
    expected_tiers = {
        "find_my_account": 0, "list_services": 0, "get_availability": 0,
        "get_my_appointments": 0, "get_quote": 0, "get_payment_link": 0,
        "book_appointment": 1, "reschedule_appointment": 1,
        "cancel_appointment": 2, "request_human_callback": 3,
    }
    assert {name: spec.tier for name, spec in REGISTRY_C.items()} == expected_tiers


def test_staff_only_tool_is_absent_from_registry_c(edge_db):
    result = dispatch(REGISTRY_C, "merge_customers", Principal(type="customer", id=1))
    assert result == {"decision": Decision.DENIED.value, "reason": Reason.NOT_IN_REGISTRY.value, "tool": "merge_customers"}


def test_list_services_excludes_archived_and_offline_only(edge_db):
    result = dispatch(REGISTRY_C, "list_services", Principal(type="customer", id=None))
    ids = {s["id"] for s in result["services"]}
    assert 10 not in ids  # archived duplicate
    assert 9 not in ids  # bookable_online=0
    assert 1 in ids and 5 in ids and 6 in ids


def test_get_quote_null_price_escalates(edge_db):
    result = dispatch(REGISTRY_C, "get_quote", Principal(type="customer", id=None), service_item_id=3)
    assert result["price_cents"] is None
    assert result["reason"] == Reason.NULL_PRICE.value


def test_get_quote_normal_price(edge_db):
    result = dispatch(REGISTRY_C, "get_quote", Principal(type="customer", id=None), service_item_id=1)
    assert result["price_cents"] == 15000
    assert result["reason"] is None


def test_get_my_appointments_scoped_to_principal(edge_db):
    result = dispatch(REGISTRY_C, "get_my_appointments", Principal(type="customer", id=1))
    assert result["decision"] == Decision.EXECUTED.value
    assert all(a["id"] in (1,) or True for a in result["appointments"])
    ids = {a["id"] for a in result["appointments"]}
    with get_session() as session:
        expected = {a.id for a in session.query(Appointment).filter(Appointment.customer_id == 1)}
    assert ids == expected


def test_get_my_appointments_rejects_mismatched_customer_id_argument(edge_db):
    result = dispatch(
        REGISTRY_C, "get_my_appointments", Principal(type="customer", id=1), customer_id=999,
    )
    assert result == {"decision": Decision.DENIED.value, "reason": Reason.PRINCIPAL_MISMATCH.value}
    with get_session() as session:
        # No appointment data for customer 999 (or anyone else) leaked into the response, and
        # the denial itself is on record.
        from db.models import AuditLog
        row = session.query(AuditLog).filter(AuditLog.tool == "get_my_appointments").one()
        assert row.reason == Reason.PRINCIPAL_MISMATCH.value


def test_get_my_appointments_unresolved_principal_declines(edge_db):
    result = dispatch(REGISTRY_C, "get_my_appointments", Principal(type="customer", id=None))
    assert result["decision"] == Decision.DENIED.value
    assert result["reason"] == Reason.UNRESOLVED_PRINCIPAL.value


def test_get_payment_link_own_only(edge_db):
    ok = dispatch(REGISTRY_C, "get_payment_link", Principal(type="customer", id=9), invoice_id=3)
    assert ok["decision"] == Decision.EXECUTED.value
    assert "payment_link" in ok

    mismatch = dispatch(REGISTRY_C, "get_payment_link", Principal(type="customer", id=13), invoice_id=3)
    assert mismatch == {"decision": Decision.DENIED.value, "reason": Reason.PRINCIPAL_MISMATCH.value}


def test_book_appointment_executes_inside_envelope(edge_db_with_policy, in_envelope_start):
    result = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=2, start_ts=in_envelope_start,
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    assert result["decision"] == Decision.EXECUTED.value
    with get_session() as session:
        appt = session.get(Appointment, result["appointment_id"])
        assert appt.customer_id == 14
        assert appt.status == "scheduled"


def test_book_appointment_queues_outside_business_hours(edge_db_with_policy, in_envelope_start):
    after_hours = in_envelope_start.replace(hour=20)
    result = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=2, start_ts=after_hours,
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    assert result["decision"] == Decision.QUEUED.value
    assert result["reason"] == Reason.OUTSIDE_BUSINESS_HOURS.value
    with get_session() as session:
        assert session.get(PendingRequest, result["request_id"]) is not None
        # nothing actually booked
        assert session.query(Appointment).filter(Appointment.customer_id == 14).count() == 1  # only the seeded one


def test_book_appointment_queues_for_balance_hold(edge_db_with_policy, in_envelope_start):
    result = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=13),
        service_item_id=1, start_ts=in_envelope_start,
        name="Harold Jennings", email="hjennings@example.com", phone="619-555-0620", address="310 Broadway",
    )
    assert result["decision"] == Decision.QUEUED.value
    assert result["reason"] == Reason.BALANCE_HOLD.value


def test_book_appointment_denies_null_price_item(edge_db, in_envelope_start):
    result = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=3, start_ts=in_envelope_start,
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    assert result["decision"] == Decision.DENIED.value
    assert result["reason"] == Reason.NULL_PRICE.value


def test_book_appointment_fall_forward_executes_below_deposit_inside_envelope(edge_db_with_policy, in_envelope_start):
    result = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=None),
        service_item_id=2, start_ts=in_envelope_start,
        name="Brand New Person", email="brandnew@example.com", phone="619-555-9999", address="1 New St",
    )
    assert result["decision"] == Decision.EXECUTED.value
    assert result["reason"] in (Reason.AMBIGUOUS_IDENTITY.value, Reason.UNRESOLVED_PRINCIPAL.value)
    with get_session() as session:
        customer = session.get(Customer, result["customer_id"])
        assert customer is not None
        assert customer.name == "Brand New Person"
        assert customer.balance_cents == 0


def test_book_appointment_fall_forward_never_denies_only_queues(edge_db_with_policy, in_envelope_start):
    """Out-of-envelope + unresolved identity queues with provisional_cap. Never DENIED —
    queueing preserves the job."""
    after_hours = in_envelope_start.replace(hour=20)
    result = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=None),
        service_item_id=2, start_ts=after_hours,
        name="Evasive Caller", email="evasive@example.com", phone="619-555-8888", address="2 Nowhere Ave",
    )
    assert result["decision"] == Decision.QUEUED.value
    assert result["reason"] == Reason.PROVISIONAL_CAP.value
    with get_session() as session:
        customer = session.get(Customer, result["customer_id"])
        assert customer is not None  # the record was still created...
        assert session.query(Appointment).filter(Appointment.customer_id == customer.id).count() == 0  # ...but nothing was booked


def test_book_appointment_needs_confirm_above_deposit_threshold(edge_db_with_policy, in_envelope_start):
    with get_session() as session:
        session.add(ServiceItem(
            id=90, name="Whole-House Repipe", base_price_cents=60000, duration_min=240,
            requires_skill="plumbing", bookable_online=1, archived=0,
        ))
        session.commit()

    pending = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=90, start_ts=in_envelope_start,
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    assert pending["decision"] == Decision.NEEDS_CONFIRM.value
    assert pending["reason"] == Reason.DEPOSIT_REQUIRED.value

    confirmed = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=90, start_ts=in_envelope_start,
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
        confirmed=True,
    )
    assert confirmed["decision"] == Decision.EXECUTED.value


def test_book_appointment_queues_when_no_skilled_technician(edge_db_with_policy, in_envelope_start):
    with get_session() as session:
        session.add(ServiceItem(
            id=91, name="Gas Line Fitting", base_price_cents=10000, duration_min=60,
            requires_skill="gas_fitting", bookable_online=1, archived=0,
        ))
        session.commit()

    result = dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=91, start_ts=in_envelope_start,
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    assert result["decision"] == Decision.QUEUED.value
    assert result["reason"] == Reason.NO_SKILLED_TECH.value


def test_cancel_appointment_needs_confirm_then_executes(edge_db_with_policy):
    """Appointment 1 is scheduled ~2 days out from the seed's fixed anchor, well inside a
    24h cancellation-fee window measured from *today* — force a near appointment directly to
    keep this deterministic regardless of when the suite runs."""
    with get_session() as session:
        soon = session.get(Appointment, 1)
        soon.customer_id = 1
        from db.seed_common import now_utc
        soon.start_ts = now_utc() + timedelta(hours=2)
        soon.end_ts = soon.start_ts + timedelta(hours=1)
        session.commit()

    pending = cancel_appointment(principal=Principal(type="customer", id=1), appointment_id=1)
    assert pending["decision"] == Decision.NEEDS_CONFIRM.value
    assert pending["reason"] == Reason.CANCELLATION_FEE.value

    done = cancel_appointment(principal=Principal(type="customer", id=1), appointment_id=1, confirmed=True)
    assert done["decision"] == Decision.EXECUTED.value
    assert done["fee_applied"] is True
    with get_session() as session:
        assert session.get(Appointment, 1).status == "cancelled"


def test_cancel_appointment_rejects_non_owner(edge_db_with_policy):
    result = cancel_appointment(principal=Principal(type="customer", id=99), appointment_id=1)
    assert result["decision"] == Decision.DENIED.value
    assert result["reason"] == Reason.PRINCIPAL_MISMATCH.value
    with get_session() as session:
        assert session.get(Appointment, 1).status == "scheduled"


def test_request_human_callback_always_queues(edge_db):
    result = dispatch(
        REGISTRY_C, "request_human_callback", Principal(type="customer", id=1),
        message="Please call me back about my furnace.",
    )
    assert result["decision"] == Decision.QUEUED.value
    with get_session() as session:
        row = session.get(PendingRequest, result["request_id"])
        assert row.tool == "request_human_callback"
        assert row.status == "pending"


def test_book_appointment_never_writes_two_audit_rows_for_one_call(edge_db_with_policy, in_envelope_start):
    """One tool call, one audit row — the in-transaction convention this whole project rests
    on, checked from the outside."""
    from db.models import AuditLog

    dispatch(
        REGISTRY_C, "book_appointment", Principal(type="customer", id=14),
        service_item_id=2, start_ts=in_envelope_start,
        name="Nancy Pham", email="npham@example.com", phone="619-555-0654", address="88 University Ave",
    )
    with get_session() as session:
        assert session.query(AuditLog).filter(AuditLog.tool == "book_appointment").count() == 1
