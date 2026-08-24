"""Registry S — the staff-facing tool surface. 18 tools, trusted but fallible: no per-field PII
masking (S sees what staff would see on any real dispatch screen), but still fail-closed on
scope (every list-style read requires an explicit filter — nothing defaults to "everything")
and still principal-scoped where ownership is meaningful (`record_payment` is manager+ only).

Registry S has no `delete_customer`, no bulk delete, no raw-SQL write helper, and no
`modify_policy_config` — Tier 4 tools that simply do not exist as functions, here or anywhere
in this codebase.

Persona S already has "a human is present to confirm" as a channel characteristic,
so unlike Registry C, most Tier-2 tools here execute immediately rather than needing a second
in-conversation confirm round-trip — the confirmation *is* the fact that a staff member issued
the command. `cancel_appointment_with_notice` is the one exception worth calling out: it
still doesn't re-ask, because staff overriding a fee window is exactly the kind of judgment
call Persona S is trusted to make that Persona C is not.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any

from sqlalchemy import func, or_

from db.database import get_session
from db.models import Appointment, Customer, Invoice, InvoiceLine, ServiceItem, Technician
from db.seed_common import UNPAID_STATUSES, now_utc, recompute_balances
from tools.audit import write_audit
from tools.dispatcher import Decision, Registry, ToolSpec
from tools.policy import check_booking_window, check_business_hours, check_discount, check_lead_time, load_policy
from tools.principal import Principal
from tools.reasons import Reason
from tools.tools_common import (
    find_available_technician,
    get_bookable_service_item,
    queue_request,
    render_diff,
    technician_has_overlap,
    technician_has_skill,
)


def _normalize(text: str | None) -> str:
    return " ".join((text or "").strip().lower().split())


def _serialize_customer(c: Customer) -> dict[str, Any]:
    return {
        "id": c.id, "name": c.name, "phone": c.phone, "email": c.email,
        "address_line": c.address_line, "city": c.city, "zip": c.zip,
        "balance_cents": c.balance_cents, "internal_notes": c.internal_notes,
        "merged_into_id": c.merged_into_id,
    }


def _serialize_appointment(a: Appointment) -> dict[str, Any]:
    return {
        "id": a.id, "customer_id": a.customer_id, "technician_id": a.technician_id,
        "service_item_id": a.service_item_id, "start_ts": a.start_ts.isoformat(),
        "end_ts": a.end_ts.isoformat(), "status": a.status,
    }


def _serialize_invoice(i: Invoice) -> dict[str, Any]:
    return {
        "id": i.id, "customer_id": i.customer_id, "appointment_id": i.appointment_id,
        "status": i.status, "total_cents": i.total_cents,
        "issued_at": i.issued_at.isoformat() if i.issued_at else None,
        "due_at": i.due_at.isoformat() if i.due_at else None,
        "paid_at": i.paid_at.isoformat() if i.paid_at else None,
        "processor_ref": i.processor_ref,
    }


# --- Tier 0: reads, every one explicitly scoped ------------------------------------------


def search_customers(*, principal: Principal, run_id: str | None = None, query: str) -> dict[str, Any]:
    args = {"query": query}
    with get_session() as session:
        if not query or not query.strip():
            write_audit(
                session, principal=principal, tool="search_customers", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value,
                     "message": "A search query is required."}

        like = f"%{query.strip().lower()}%"
        rows = (
            session.query(Customer)
            .filter(or_(
                func.lower(Customer.name).like(like),
                func.lower(Customer.phone).like(like),
                func.lower(Customer.email).like(like),
                func.lower(Customer.address_line).like(like),
            ))
            .order_by(Customer.id)
            .all()
        )
        data = [_serialize_customer(c) for c in rows]
        write_audit(
            session, principal=principal, tool="search_customers", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "customers": data}


def get_customer_detail(*, principal: Principal, run_id: str | None = None, customer_id: int) -> dict[str, Any]:
    args = {"customer_id": customer_id}
    with get_session() as session:
        customer = session.get(Customer, customer_id)
        if customer is None:
            write_audit(
                session, principal=principal, tool="get_customer_detail", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        write_audit(
            session, principal=principal, tool="get_customer_detail", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"customer:{customer_id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "customer": _serialize_customer(customer)}


def list_appointments(
    *, principal: Principal, run_id: str | None = None,
    range_start: datetime, range_end: datetime,
    customer_id: int | None = None, technician_id: int | None = None,
) -> dict[str, Any]:
    """range_start/range_end are mandatory — a bounded window is the explicit scope this list
    requires; there is no unscoped 'list every appointment' form."""
    args = {"range_start": range_start, "range_end": range_end, "customer_id": customer_id, "technician_id": technician_id}
    with get_session() as session:
        query = session.query(Appointment).filter(
            Appointment.start_ts >= range_start, Appointment.start_ts < range_end,
        )
        if customer_id is not None:
            query = query.filter(Appointment.customer_id == customer_id)
        if technician_id is not None:
            query = query.filter(Appointment.technician_id == technician_id)
        rows = query.order_by(Appointment.start_ts).all()
        data = [_serialize_appointment(a) for a in rows]
        write_audit(
            session, principal=principal, tool="list_appointments", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "appointments": data}


def get_schedule(*, principal: Principal, run_id: str | None = None, technician_id: int, date: datetime) -> dict[str, Any]:
    args = {"technician_id": technician_id, "date": date}
    with get_session() as session:
        tech = session.get(Technician, technician_id)
        if tech is None:
            write_audit(
                session, principal=principal, tool="get_schedule", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        day_start = datetime(date.year, date.month, date.day)
        day_end = day_start + timedelta(days=1)
        rows = (
            session.query(Appointment)
            .filter(Appointment.technician_id == technician_id, Appointment.start_ts >= day_start, Appointment.start_ts < day_end)
            .order_by(Appointment.start_ts)
            .all()
        )
        data = [_serialize_appointment(a) for a in rows]
        write_audit(
            session, principal=principal, tool="get_schedule", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"technician:{technician_id}", run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "technician_id": technician_id, "appointments": data}


def list_invoices(
    *, principal: Principal, run_id: str | None = None,
    customer_id: int | None = None, status: str | None = None,
) -> dict[str, Any]:
    """At least one of customer_id/status is required — same fail-closed rule as every other
    list tool in this project."""
    args = {"customer_id": customer_id, "status": status}
    if customer_id is None and status is None:
        with get_session() as session:
            write_audit(
                session, principal=principal, tool="list_invoices", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
        return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value,
                 "message": "Provide at least a customer_id or a status to scope this list."}

    with get_session() as session:
        query = session.query(Invoice)
        if customer_id is not None:
            query = query.filter(Invoice.customer_id == customer_id)
        if status is not None:
            query = query.filter(Invoice.status == status)
        rows = query.order_by(Invoice.id).all()
        data = [_serialize_invoice(i) for i in rows]
        write_audit(
            session, principal=principal, tool="list_invoices", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "invoices": data}


def find_duplicate_candidates(*, principal: Principal, run_id: str | None = None, min_score: float = 0.82) -> dict[str, Any]:
    """A DB-wide scan by design — 'find duplicates across the customer base' is itself the
    explicit scope, unlike an unfiltered 'list every customer.'"""
    args = {"min_score": min_score}
    with get_session() as session:
        customers = session.query(Customer).filter(Customer.merged_into_id.is_(None)).order_by(Customer.id).all()
        results: list[dict[str, Any]] = []
        for a, b in combinations(customers, 2):
            signals: list[str] = []
            name_ratio = SequenceMatcher(None, _normalize(a.name), _normalize(b.name)).ratio()
            same_phone = bool(a.phone) and a.phone == b.phone
            same_address = (
                _normalize(a.address_line) == _normalize(b.address_line) != ""
                and _normalize(a.city) == _normalize(b.city)
            )
            if same_phone:
                signals.append("shared_phone")
            if name_ratio >= min_score:
                signals.append(f"similar_name:{name_ratio:.2f}")
            if same_address:
                signals.append("same_address")
            if not signals:
                continue
            score = max([name_ratio] + [1.0 for s in signals if s in ("shared_phone", "same_address")])
            results.append({
                "customer_id_a": a.id, "customer_id_b": b.id,
                "score": round(score, 2), "signals": signals,
            })
        results.sort(key=lambda r: -r["score"])
        write_audit(
            session, principal=principal, tool="find_duplicate_candidates", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "candidates": results}


def find_schedule_conflicts(*, principal: Principal, run_id: str | None = None, range_start: datetime, range_end: datetime) -> dict[str, Any]:
    args = {"range_start": range_start, "range_end": range_end}
    with get_session() as session:
        appts = (
            session.query(Appointment)
            .filter(
                Appointment.status == "scheduled",
                Appointment.technician_id.isnot(None),
                Appointment.start_ts < range_end,
                Appointment.end_ts > range_start,
            )
            .order_by(Appointment.technician_id, Appointment.start_ts)
            .all()
        )
        by_tech: dict[int, list[Appointment]] = {}
        for appt in appts:
            by_tech.setdefault(appt.technician_id, []).append(appt)

        conflicts = []
        for tech_id, lst in by_tech.items():
            for a, b in combinations(lst, 2):
                if a.start_ts < b.end_ts and b.start_ts < a.end_ts:
                    conflicts.append({"technician_id": tech_id, "appointment_id_a": a.id, "appointment_id_b": b.id})

        write_audit(
            session, principal=principal, tool="find_schedule_conflicts", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "conflicts": conflicts}


# --- Tier 1: autonomous writes -------------------------------------------------------------


def book_appointment_for_customer(
    *, principal: Principal, run_id: str | None = None,
    customer_id: int, service_item_id: int, start_ts: datetime,
    override_business_hours: bool = False, override_online_bookable: bool = False,
) -> dict[str, Any]:
    """Staff may override the business-hours and online-bookable checks explicitly; lead
    time, booking window, and technician skill/availability are not overridable here — those
    are physical constraints, not business-policy preferences. Unlike book_appointment (C),
    an unmet requirement is a direct denial: staff IS the human this would otherwise escalate
    to, so there's nobody further to queue it for."""
    args = {
        "customer_id": customer_id, "service_item_id": service_item_id, "start_ts": start_ts,
        "override_business_hours": override_business_hours, "override_online_bookable": override_online_bookable,
    }
    with get_session() as session:
        customer = session.get(Customer, customer_id)
        item = get_bookable_service_item(session, service_item_id)
        if customer is None or item is None:
            write_audit(
                session, principal=principal, tool="book_appointment_for_customer", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        entity_ref = f"customer:{customer_id}"
        if item.base_price_cents is None:
            write_audit(
                session, principal=principal, tool="book_appointment_for_customer", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.NULL_PRICE.value,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.NULL_PRICE.value}

        policy = load_policy(session)
        now = now_utc()
        end_ts = start_ts + timedelta(minutes=item.duration_min)
        tech = find_available_technician(session, item, start_ts, end_ts)
        hours_ok, _ = check_business_hours(policy, start_ts)
        lead_ok, lead_reason = check_lead_time(policy, start_ts, now)
        window_ok, window_reason = check_booking_window(policy, start_ts, now)
        checks = [
            (bool(item.bookable_online) or override_online_bookable, Reason.NOT_ONLINE_BOOKABLE.value),
            (hours_ok or override_business_hours, Reason.OUTSIDE_BUSINESS_HOURS.value),
            (lead_ok, lead_reason),
            (window_ok, window_reason),
            (tech is not None, Reason.NO_SKILLED_TECH.value),
        ]
        first_fail = next((code for ok, code in checks if not ok), None)
        if first_fail is not None:
            write_audit(
                session, principal=principal, tool="book_appointment_for_customer", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=first_fail, entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": first_fail}

        appt = Appointment(
            customer_id=customer_id, technician_id=tech.id, service_item_id=item.id,
            start_ts=start_ts, end_ts=end_ts, status="scheduled",
            created_by=f"staff:{principal.id}", created_via="agent_s",
        )
        session.add(appt)
        session.flush()
        write_audit(
            session, principal=principal, tool="book_appointment_for_customer", declared_tier=1,
            decision=Decision.EXECUTED.value, args=args, entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "appointment_id": appt.id, "technician_id": tech.id}


def reassign_technician(*, principal: Principal, run_id: str | None = None, appointment_id: int, technician_id: int) -> dict[str, Any]:
    """Staff can knowingly override skill/availability — that judgment is exactly what this
    tool is for. What it must never do is hide the fact that it did: a mismatch or a new
    double-booking is reported back, not silently absorbed."""
    args = {"appointment_id": appointment_id, "technician_id": technician_id}
    with get_session() as session:
        appt = session.get(Appointment, appointment_id)
        tech = session.get(Technician, technician_id)
        if appt is None or tech is None:
            write_audit(
                session, principal=principal, tool="reassign_technician", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        item = session.get(ServiceItem, appt.service_item_id)
        skill_ok = technician_has_skill(tech, item.requires_skill) if item else True
        overlap = technician_has_overlap(session, technician_id, appt.start_ts, appt.end_ts, exclude_appointment_id=appt.id)

        appt.technician_id = technician_id
        entity_ref = f"appointment:{appointment_id}"
        write_audit(
            session, principal=principal, tool="reassign_technician", declared_tier=1,
            decision=Decision.EXECUTED.value, args=args,
            reason=Reason.NO_SKILLED_TECH.value if not skill_ok else None,
            entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        warnings = []
        if not skill_ok:
            warnings.append("technician lacks the skill this service requires")
        if overlap:
            warnings.append("technician is now double-booked at this time")
        return {"decision": Decision.EXECUTED.value, "appointment_id": appointment_id, "technician_id": technician_id, "warnings": warnings}


def add_internal_note(*, principal: Principal, run_id: str | None = None, customer_id: int, note_text: str) -> dict[str, Any]:
    args = {"customer_id": customer_id, "note_text": note_text}
    with get_session() as session:
        customer = session.get(Customer, customer_id)
        if customer is None:
            write_audit(
                session, principal=principal, tool="add_internal_note", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        stamp = now_utc().isoformat(timespec="minutes")
        entry = f"[{stamp} staff:{principal.id}] {note_text}"
        customer.internal_notes = f"{customer.internal_notes}\n{entry}" if customer.internal_notes else entry
        write_audit(
            session, principal=principal, tool="add_internal_note", declared_tier=1,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"customer:{customer_id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "customer_id": customer_id}


# --- Tier 2: immediate writes (the staff channel's "human present" already is the confirm) --


def create_invoice(
    *, principal: Principal, run_id: str | None = None,
    customer_id: int, line_items: list[dict[str, Any]], appointment_id: int | None = None,
) -> dict[str, Any]:
    args = {"customer_id": customer_id, "line_items": line_items, "appointment_id": appointment_id}
    with get_session() as session:
        customer = session.get(Customer, customer_id)
        if customer is None or not line_items:
            write_audit(
                session, principal=principal, tool="create_invoice", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        invoice = Invoice(customer_id=customer_id, appointment_id=appointment_id, status="draft", total_cents=0)
        session.add(invoice)
        session.flush()
        total = 0
        for line in line_items:
            qty = int(line.get("qty", 1))
            unit_price_cents = int(line["unit_price_cents"])
            total += qty * unit_price_cents
            session.add(InvoiceLine(
                invoice_id=invoice.id, service_item_id=line.get("service_item_id"),
                description=line.get("description"), qty=qty, unit_price_cents=unit_price_cents,
            ))
        invoice.total_cents = total
        recompute_balances(session)
        write_audit(
            session, principal=principal, tool="create_invoice", declared_tier=2,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"invoice:{invoice.id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "invoice_id": invoice.id, "total_cents": total}


def send_invoice(*, principal: Principal, run_id: str | None = None, invoice_id: int) -> dict[str, Any]:
    args = {"invoice_id": invoice_id}
    with get_session() as session:
        invoice = session.get(Invoice, invoice_id)
        if invoice is None or invoice.status != "draft":
            write_audit(
                session, principal=principal, tool="send_invoice", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value,
                     "message": "Only a draft invoice can be sent."}

        now = now_utc()
        invoice.status = "sent"
        invoice.issued_at = invoice.issued_at or now
        invoice.due_at = invoice.due_at or (now + timedelta(days=14))
        write_audit(
            session, principal=principal, tool="send_invoice", declared_tier=2,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"invoice:{invoice_id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "invoice_id": invoice_id, "status": "sent"}


def _execute_apply_discount(session, invoice: Invoice, discount_pct: int) -> dict[str, Any]:
    """The actual discount write, shared by the ≤cap immediate path and approve.py's
    post-approval execution of an over-cap request."""
    discount_amount = round(invoice.total_cents * discount_pct / 100)
    session.add(InvoiceLine(
        invoice_id=invoice.id, service_item_id=None,
        description=f"Discount ({discount_pct}%)", qty=1, unit_price_cents=-discount_amount,
    ))
    invoice.total_cents -= discount_amount
    recompute_balances(session)
    return {"invoice_id": invoice.id, "discount_pct": discount_pct, "new_total_cents": invoice.total_cents}


def apply_discount(*, principal: Principal, run_id: str | None = None, invoice_id: int, discount_pct: int) -> dict[str, Any]:
    """Declared Tier 2. Runtime escalates to QUEUED, never below, when discount_pct exceeds
    the policy cap — the same escalate-never-de-escalate pattern as book_appointment."""
    args = {"invoice_id": invoice_id, "discount_pct": discount_pct}
    with get_session() as session:
        invoice = session.get(Invoice, invoice_id)
        if invoice is None or invoice.status not in ("draft", "sent"):
            write_audit(
                session, principal=principal, tool="apply_discount", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        policy = load_policy(session)
        ok, reason = check_discount(policy, discount_pct)
        entity_ref = f"invoice:{invoice_id}"

        if ok:
            result = _execute_apply_discount(session, invoice, discount_pct)
            write_audit(
                session, principal=principal, tool="apply_discount", declared_tier=2,
                decision=Decision.EXECUTED.value, args=args, entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.EXECUTED.value, **result}

        projected_total = invoice.total_cents - round(invoice.total_cents * discount_pct / 100)
        preview = render_diff(entity_ref, before={"total_cents": invoice.total_cents}, after={"total_cents": projected_total})
        pending = queue_request(session, principal=principal, tool="apply_discount", args=args, preview_text=preview)
        write_audit(
            session, principal=principal, tool="apply_discount", declared_tier=2,
            decision=Decision.QUEUED.value, args=args, reason=reason, entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.QUEUED.value, "request_id": pending.id, "reason": reason, "preview_text": pending.preview_text}


def cancel_appointment_with_notice(*, principal: Principal, run_id: str | None = None, appointment_id: int, notice_text: str) -> dict[str, Any]:
    args = {"appointment_id": appointment_id, "notice_text": notice_text}
    with get_session() as session:
        appt = session.get(Appointment, appointment_id)
        if appt is None or appt.status != "scheduled":
            write_audit(
                session, principal=principal, tool="cancel_appointment_with_notice", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        appt.status = "cancelled"
        write_audit(
            session, principal=principal, tool="cancel_appointment_with_notice", declared_tier=2,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"appointment:{appointment_id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "appointment_id": appointment_id}


def record_payment(*, principal: Principal, run_id: str | None = None, invoice_id: int, processor_ref: str, amount_cents: int) -> dict[str, Any]:
    """Manager+ only (enforced by the registry's min_role, re-checked by the dispatcher before
    this function ever runs). Payment is reconciled against a processor reference, never
    asserted — and never partial in v1: amount must match the invoice total exactly."""
    args = {"invoice_id": invoice_id, "processor_ref": processor_ref, "amount_cents": amount_cents}
    with get_session() as session:
        invoice = session.get(Invoice, invoice_id)
        if invoice is None or invoice.status not in ("draft", "sent"):
            write_audit(
                session, principal=principal, tool="record_payment", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        if amount_cents != invoice.total_cents:
            write_audit(
                session, principal=principal, tool="record_payment", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value,
                entity_ref=f"invoice:{invoice_id}", run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value,
                     "message": "Partial payments are not supported; amount must match the invoice total."}

        invoice.status = "paid"
        invoice.paid_at = now_utc()
        invoice.processor_ref = processor_ref
        recompute_balances(session)
        write_audit(
            session, principal=principal, tool="record_payment", declared_tier=2,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"invoice:{invoice_id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "invoice_id": invoice_id, "status": "paid"}


# --- Tier 3: deferred approval only — these never execute autonomously -------------------


def _execute_write_off_balance(session, customer_id: int) -> dict[str, Any]:
    unpaid = session.query(Invoice).filter(Invoice.customer_id == customer_id, Invoice.status.in_(UNPAID_STATUSES)).all()
    for invoice in unpaid:
        invoice.status = "void"
    recompute_balances(session)
    return {"customer_id": customer_id, "invoices_voided": [i.id for i in unpaid]}


def write_off_balance(*, principal: Principal, run_id: str | None = None, customer_id: int, note: str) -> dict[str, Any]:
    args = {"customer_id": customer_id, "note": note}
    with get_session() as session:
        customer = session.get(Customer, customer_id)
        if customer is None:
            write_audit(
                session, principal=principal, tool="write_off_balance", declared_tier=3,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        entity_ref = f"customer:{customer_id}"
        preview = render_diff(entity_ref, before={"balance_cents": customer.balance_cents}, after={"balance_cents": 0})
        pending = queue_request(session, principal=principal, tool="write_off_balance", args=args, preview_text=preview)
        write_audit(
            session, principal=principal, tool="write_off_balance", declared_tier=3,
            decision=Decision.QUEUED.value, args=args, entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.QUEUED.value, "request_id": pending.id, "preview_text": pending.preview_text}


def _execute_void_invoice(session, invoice_id: int) -> dict[str, Any]:
    invoice = session.get(Invoice, invoice_id)
    invoice.status = "void"
    recompute_balances(session)
    return {"invoice_id": invoice_id, "status": "void"}


def void_invoice(*, principal: Principal, run_id: str | None = None, invoice_id: int) -> dict[str, Any]:
    args = {"invoice_id": invoice_id}
    with get_session() as session:
        invoice = session.get(Invoice, invoice_id)
        if invoice is None or invoice.status == "void":
            write_audit(
                session, principal=principal, tool="void_invoice", declared_tier=3,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        entity_ref = f"invoice:{invoice_id}"
        preview = render_diff(entity_ref, before={"status": invoice.status}, after={"status": "void"})
        pending = queue_request(session, principal=principal, tool="void_invoice", args=args, preview_text=preview)
        write_audit(
            session, principal=principal, tool="void_invoice", declared_tier=3,
            decision=Decision.QUEUED.value, args=args, entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.QUEUED.value, "request_id": pending.id, "preview_text": pending.preview_text}


def _execute_merge_customers(session, survivor_id: int, loser_id: int) -> dict[str, Any]:
    """Soft merge only: the loser's own rows (appointments, invoices) stay attached to its own
    customer_id — merged_into_id is a pointer, not a data migration, which is what makes this
    reversible without a rebuild."""
    loser = session.get(Customer, loser_id)
    loser.merged_into_id = survivor_id
    return {"survivor_id": survivor_id, "loser_id": loser_id}


def merge_customers(*, principal: Principal, run_id: str | None = None, survivor_id: int, loser_id: int) -> dict[str, Any]:
    """Second-human sign-off: this always queues, with a rendered field-by-field diff between
    the two records so the approver reviews an actual comparison, not a function call."""
    args = {"survivor_id": survivor_id, "loser_id": loser_id}
    with get_session() as session:
        survivor = session.get(Customer, survivor_id)
        loser = session.get(Customer, loser_id)
        if survivor is None or loser is None or survivor_id == loser_id:
            write_audit(
                session, principal=principal, tool="merge_customers", declared_tier=3,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        preview = render_diff(
            f"customer:{loser_id} -> customer:{survivor_id}",
            before={"name": loser.name, "phone": loser.phone, "email": loser.email, "address": loser.address_line},
            after={"name": survivor.name, "phone": survivor.phone, "email": survivor.email, "address": survivor.address_line},
        )
        pending = queue_request(session, principal=principal, tool="merge_customers", args=args, preview_text=preview)
        write_audit(
            session, principal=principal, tool="merge_customers", declared_tier=3,
            decision=Decision.QUEUED.value, args=args, entity_ref=f"customer:{loser_id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.QUEUED.value, "request_id": pending.id, "preview_text": pending.preview_text}


REGISTRY_S: Registry = {
    "search_customers": ToolSpec(fn=search_customers, tier=0),
    "get_customer_detail": ToolSpec(fn=get_customer_detail, tier=0),
    "list_appointments": ToolSpec(fn=list_appointments, tier=0),
    "get_schedule": ToolSpec(fn=get_schedule, tier=0),
    "list_invoices": ToolSpec(fn=list_invoices, tier=0),
    "find_duplicate_candidates": ToolSpec(fn=find_duplicate_candidates, tier=0),
    "find_schedule_conflicts": ToolSpec(fn=find_schedule_conflicts, tier=0),
    "book_appointment_for_customer": ToolSpec(fn=book_appointment_for_customer, tier=1),
    "reassign_technician": ToolSpec(fn=reassign_technician, tier=1),
    "add_internal_note": ToolSpec(fn=add_internal_note, tier=1),
    "create_invoice": ToolSpec(fn=create_invoice, tier=2),
    "send_invoice": ToolSpec(fn=send_invoice, tier=2),
    "apply_discount": ToolSpec(fn=apply_discount, tier=2),
    "cancel_appointment_with_notice": ToolSpec(fn=cancel_appointment_with_notice, tier=2),
    "record_payment": ToolSpec(fn=record_payment, tier=2, min_role="manager"),
    "write_off_balance": ToolSpec(fn=write_off_balance, tier=3),
    "void_invoice": ToolSpec(fn=void_invoice, tier=3),
    "merge_customers": ToolSpec(fn=merge_customers, tier=3),
}

assert len(REGISTRY_S) == 18, "Registry S must have exactly 18 tools"
