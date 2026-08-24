"""Registry C — the customer-facing tool surface. 10 tools, exactly what's in the table
below and nothing else. There is no customer-creation tool, no cross-customer search, no
`internal_notes` or `balance_cents` anywhere in a return value, no invoice creation, no price
override, no technician reassignment, no merge, no analytics, no notification tool — their
absence from this file *is* the authorization boundary for Persona C, not a prompt telling the
model not to use them.

Every tool function has the same shape: open one Session, do reads/writes, write exactly one
audit_log row, one commit — or nothing commits at all.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any

from db.database import get_session
from db.models import Appointment, Customer, Invoice, ServiceItem
from db.seed_common import now_utc
from tools.dispatcher import Decision, Registry, ToolSpec
from tools.identity import UNRESOLVED, find_my_account, resolve_candidates
from tools.policy import (
    cancellation_fee_applies,
    check_balance_hold,
    check_booking_window,
    check_business_hours,
    check_lead_time,
    deposit_required,
    load_policy,
)
from tools.principal import Principal
from tools.reasons import Reason
from tools.tools_common import find_available_technician, get_bookable_service_item, queue_request, render_diff
from tools.audit import write_audit

_MAX_AVAILABILITY_SLOTS = 40
_AVAILABILITY_STEP_MIN = 30


def _serialize_appointment(appt: Appointment) -> dict[str, Any]:
    return {
        "id": appt.id,
        "service_item_id": appt.service_item_id,
        "technician_id": appt.technician_id,
        "start_ts": appt.start_ts.isoformat(),
        "end_ts": appt.end_ts.isoformat(),
        "status": appt.status,
    }


# --- Tier 0: reads ---------------------------------------------------------------------


def find_my_account_tool(
    *, principal: Principal, run_id: str | None = None,
    name: str, email: str, phone: str, address: str,
) -> dict[str, Any]:
    """The one and only identity-collection entry point. Full tuple, called once — see
    tools/identity.py for why the shape of what this returns can never leak more than
    resolved/unresolved. `customer_id` in the return value is for the harness to update its
    own session-level principal with (out-of-band, never re-submitted by the model as an
    argument to anything) — not a capability grant by itself."""
    args = {"name": name, "email": email, "phone": phone, "address": address}
    with get_session() as session:
        result = find_my_account(session, name=name, email=email, phone=phone, address=address)
        resolved = result is not UNRESOLVED
        entity_ref = f"customer:{result.id}" if resolved else None
        write_audit(
            session, principal=principal, tool="find_my_account", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        if resolved:
            return {"decision": Decision.EXECUTED.value, "resolved": True, "customer_id": result.id}
        return {"decision": Decision.EXECUTED.value, "resolved": False}


def list_services(*, principal: Principal, run_id: str | None = None) -> dict[str, Any]:
    """General information — no principal gate (published prices need no identity check)."""
    with get_session() as session:
        items = (
            session.query(ServiceItem)
            .filter(ServiceItem.archived == 0, ServiceItem.bookable_online == 1)
            .order_by(ServiceItem.id)
            .all()
        )
        data = [
            {
                "id": i.id, "name": i.name, "description": i.description,
                "price_cents": i.base_price_cents, "duration_min": i.duration_min,
            }
            for i in items
        ]
        write_audit(
            session, principal=principal, tool="list_services", declared_tier=0,
            decision=Decision.EXECUTED.value, args={}, run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "services": data}


def get_availability(
    *, principal: Principal, run_id: str | None = None,
    service_item_id: int, range_start: datetime, range_end: datetime,
) -> dict[str, Any]:
    """Computed, not stored: business hours ∩ free skilled active technician ∩ lead time ∩
    booking window, sampled at a fixed granularity. General information — no principal gate."""
    args = {"service_item_id": service_item_id, "range_start": range_start, "range_end": range_end}
    with get_session() as session:
        policy = load_policy(session)
        item = get_bookable_service_item(session, service_item_id)
        if item is None or not item.bookable_online:
            reason = Reason.NOT_ONLINE_BOOKABLE.value if item else Reason.INVALID_ARGUMENT.value
            write_audit(
                session, principal=principal, tool="get_availability", declared_tier=0,
                decision=Decision.EXECUTED.value, args=args, reason=reason, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.EXECUTED.value, "slots": [], "reason": reason}

        now = now_utc()
        earliest = max(range_start, now + timedelta(hours=policy.min_lead_time_hours))
        latest = min(range_end, now + timedelta(days=policy.max_booking_window_days))

        slots: list[datetime] = []
        day = earliest.date()
        while day <= latest.date() and len(slots) < _MAX_AVAILABILITY_SLOTS:
            day_start = datetime.combine(day, time.min)
            minute = 0
            while minute < 24 * 60 and len(slots) < _MAX_AVAILABILITY_SLOTS:
                candidate = day_start + timedelta(minutes=minute)
                minute += _AVAILABILITY_STEP_MIN
                if candidate < earliest or candidate > latest:
                    continue
                hours_ok, _ = check_business_hours(policy, candidate)
                if not hours_ok:
                    continue
                candidate_end = candidate + timedelta(minutes=item.duration_min)
                if find_available_technician(session, item, candidate, candidate_end) is None:
                    continue
                slots.append(candidate)
            day += timedelta(days=1)

        write_audit(
            session, principal=principal, tool="get_availability", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "slots": [s.isoformat() for s in slots]}


def get_my_appointments(
    *, principal: Principal, run_id: str | None = None, customer_id: int | None = None
) -> dict[str, Any]:
    """No argument selects whose appointments — 'my' means the injected principal, always.
    `customer_id` exists only so a model that hallucinates the parameter gets a clean denial
    instead of a silent scope change or a TypeError if a model hallucinates the argument."""
    args = {"customer_id": customer_id}
    with get_session() as session:
        if customer_id is not None and customer_id != principal.id:
            write_audit(
                session, principal=principal, tool="get_my_appointments", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=Reason.PRINCIPAL_MISMATCH.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.PRINCIPAL_MISMATCH.value}

        if principal.id is None:
            write_audit(
                session, principal=principal, tool="get_my_appointments", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=Reason.UNRESOLVED_PRINCIPAL.value, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.DENIED.value, "reason": Reason.UNRESOLVED_PRINCIPAL.value,
                "message": "We couldn't verify your account. Please request a callback.",
            }

        appts = (
            session.query(Appointment)
            .filter(Appointment.customer_id == principal.id)
            .order_by(Appointment.start_ts)
            .all()
        )
        data = [_serialize_appointment(a) for a in appts]
        write_audit(
            session, principal=principal, tool="get_my_appointments", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"customer:{principal.id}", run_id=run_id,
        )
        session.commit()
    return {"decision": Decision.EXECUTED.value, "appointments": data}


def get_quote(*, principal: Principal, run_id: str | None = None, service_item_id: int) -> dict[str, Any]:
    """Published price lookup — general information, no principal gate. A null price must
    escalate, never be estimated."""
    args = {"service_item_id": service_item_id}
    with get_session() as session:
        item = get_bookable_service_item(session, service_item_id)
        if item is None:
            write_audit(
                session, principal=principal, tool="get_quote", declared_tier=0,
                decision=Decision.EXECUTED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.EXECUTED.value, "price_cents": None, "reason": Reason.INVALID_ARGUMENT.value}

        reason = Reason.NULL_PRICE.value if item.base_price_cents is None else None
        write_audit(
            session, principal=principal, tool="get_quote", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, reason=reason, run_id=run_id,
        )
        session.commit()
        return {
            "decision": Decision.EXECUTED.value, "service_item_id": item.id,
            "price_cents": item.base_price_cents, "reason": reason,
        }


def get_payment_link(*, principal: Principal, run_id: str | None = None, invoice_id: int) -> dict[str, Any]:
    """Own invoices only."""
    args = {"invoice_id": invoice_id}
    with get_session() as session:
        invoice = session.get(Invoice, invoice_id)
        if invoice is None:
            write_audit(
                session, principal=principal, tool="get_payment_link", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        if principal.id is None or invoice.customer_id != principal.id:
            reason = Reason.UNRESOLVED_PRINCIPAL.value if principal.id is None else Reason.PRINCIPAL_MISMATCH.value
            write_audit(
                session, principal=principal, tool="get_payment_link", declared_tier=0,
                decision=Decision.DENIED.value, args=args, reason=reason,
                entity_ref=f"invoice:{invoice_id}", run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": reason}

        write_audit(
            session, principal=principal, tool="get_payment_link", declared_tier=0,
            decision=Decision.EXECUTED.value, args=args, entity_ref=f"invoice:{invoice_id}", run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "payment_link": f"https://pay.example.test/invoices/{invoice.id}"}


# --- Tier 1: autonomous writes inside the envelope --------------------------------------


def book_appointment(
    *, principal: Principal, run_id: str | None = None,
    service_item_id: int, start_ts: datetime,
    name: str, email: str, phone: str, address: str,
    confirmed: bool = False,
) -> dict[str, Any]:
    """The flagship tool: fall-forward customer creation, automatic envelope escalation, and
    the credit-hold-bypass closure all live here, not in the agent's judgment.

    `name`/`email`/`phone`/`address` are the already-collected identity tuple. When
    `principal.id` is not None they're informational only (the harness already resolved who
    this is); when it is None, they're what gets written into the new customer row if this
    call falls forward.
    """
    args = {
        "service_item_id": service_item_id, "start_ts": start_ts,
        "name": name, "email": email, "phone": phone, "address": address, "confirmed": confirmed,
    }
    with get_session() as session:
        item = get_bookable_service_item(session, service_item_id)
        if item is None:
            write_audit(
                session, principal=principal, tool="book_appointment", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value,
                outcome="error", run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value,
                    "message": "That service isn't available to book."}

        if item.base_price_cents is None:
            write_audit(
                session, principal=principal, tool="book_appointment", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.NULL_PRICE.value, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.DENIED.value, "reason": Reason.NULL_PRICE.value,
                "message": "This service needs an inspection before it can be priced or booked. "
                           "Please request a callback.",
            }

        policy = load_policy(session)
        now = now_utc()
        end_ts = start_ts + timedelta(minutes=item.duration_min)
        tech = find_available_technician(session, item, start_ts, end_ts)

        hours_ok, hours_reason = check_business_hours(policy, start_ts)
        lead_ok, lead_reason = check_lead_time(policy, start_ts, now)
        window_ok, window_reason = check_booking_window(policy, start_ts, now)
        checks: list[tuple[bool, str]] = [
            (bool(item.bookable_online), Reason.NOT_ONLINE_BOOKABLE.value),
            (policy.auto_book_enabled, Reason.AUTO_BOOK_DISABLED.value),
            (hours_ok, hours_reason),
            (lead_ok, lead_reason),
            (window_ok, window_reason),
            (tech is not None, Reason.NO_SKILLED_TECH.value),
        ]
        first_envelope_fail = next((code for ok, code in checks if not ok), None)
        envelope_ok = first_envelope_fail is None

        creating_new_customer = principal.id is None
        if creating_new_customer:
            candidates = resolve_candidates(session, name=name, email=email, phone=phone, address=address)
            creation_reason = (
                Reason.AMBIGUOUS_IDENTITY.value if len(candidates) >= 2 else Reason.UNRESOLVED_PRINCIPAL.value
            )
            customer = Customer(
                name=name, phone=phone, email=email, address_line=address,
                balance_cents=0, created_at=now,
            )
            session.add(customer)
            session.flush()
            customer_id = customer.id
        else:
            customer = session.get(Customer, principal.id)
            while customer.merged_into_id is not None:
                customer = session.get(Customer, customer.merged_into_id)
            customer_id = customer.id
            creation_reason = None

        entity_ref = f"customer:{customer_id}"

        if creating_new_customer:
            # Provisional customers may book autonomously only inside the envelope AND
            # below deposit_required_above. Anything else queues — never denied, the job is
            # never lost just because identity didn't resolve.
            if envelope_ok and not deposit_required(policy, item.base_price_cents):
                appt = Appointment(
                    customer_id=customer_id, technician_id=tech.id, service_item_id=item.id,
                    start_ts=start_ts, end_ts=end_ts, status="scheduled",
                    created_by="agent_c", created_via="agent_c",
                )
                session.add(appt)
                session.flush()
                write_audit(
                    session, principal=principal, tool="book_appointment", declared_tier=1,
                    decision=Decision.EXECUTED.value, args=args, reason=creation_reason,
                    entity_ref=entity_ref, run_id=run_id,
                )
                session.commit()
                return {
                    "decision": Decision.EXECUTED.value, "appointment_id": appt.id,
                    "customer_id": customer_id, "technician_id": tech.id, "reason": creation_reason,
                }

            preview = render_diff(
                f"new customer '{name}' + appointment",
                before={}, after={"service": item.name, "start_ts": str(start_ts)},
            )
            pending = queue_request(
                session, principal=principal, tool="book_appointment", args=args, preview_text=preview,
            )
            write_audit(
                session, principal=principal, tool="book_appointment", declared_tier=1,
                decision=Decision.QUEUED.value, args=args, reason=Reason.PROVISIONAL_CAP.value,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.QUEUED.value, "request_id": pending.id, "customer_id": customer_id,
                "reason": Reason.PROVISIONAL_CAP.value, "preview_text": pending.preview_text,
            }

        # Resolved, pre-existing customer.
        balance_ok, balance_reason = check_balance_hold(policy, customer.balance_cents)

        if not envelope_ok:
            preview = render_diff(
                f"customer:{customer_id} + appointment",
                before={}, after={"service": item.name, "start_ts": str(start_ts)},
            )
            pending = queue_request(
                session, principal=principal, tool="book_appointment", args=args, preview_text=preview,
            )
            write_audit(
                session, principal=principal, tool="book_appointment", declared_tier=1,
                decision=Decision.QUEUED.value, args=args, reason=first_envelope_fail,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.QUEUED.value, "request_id": pending.id,
                "reason": first_envelope_fail, "preview_text": pending.preview_text,
            }

        if not balance_ok:
            preview = render_diff(
                f"customer:{customer_id} + appointment",
                before={}, after={"service": item.name, "start_ts": str(start_ts)},
            )
            pending = queue_request(
                session, principal=principal, tool="book_appointment", args=args, preview_text=preview,
            )
            write_audit(
                session, principal=principal, tool="book_appointment", declared_tier=1,
                decision=Decision.QUEUED.value, args=args, reason=balance_reason,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.QUEUED.value, "request_id": pending.id,
                "reason": balance_reason, "preview_text": pending.preview_text,
            }

        if deposit_required(policy, item.base_price_cents) and not confirmed:
            write_audit(
                session, principal=principal, tool="book_appointment", declared_tier=1,
                decision=Decision.NEEDS_CONFIRM.value, args=args, reason=Reason.DEPOSIT_REQUIRED.value,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.NEEDS_CONFIRM.value, "reason": Reason.DEPOSIT_REQUIRED.value,
                "price_cents": item.base_price_cents, "deposit_required_above_cents": policy.deposit_required_above,
                "message": "This booking requires a deposit. Confirm to proceed.",
            }

        appt = Appointment(
            customer_id=customer_id, technician_id=tech.id, service_item_id=item.id,
            start_ts=start_ts, end_ts=end_ts, status="scheduled",
            created_by="agent_c", created_via="agent_c",
        )
        session.add(appt)
        session.flush()
        write_audit(
            session, principal=principal, tool="book_appointment", declared_tier=1,
            decision=Decision.EXECUTED.value, args=args, entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {
            "decision": Decision.EXECUTED.value, "appointment_id": appt.id,
            "customer_id": customer_id, "technician_id": tech.id,
        }


def reschedule_appointment(
    *, principal: Principal, run_id: str | None = None,
    appointment_id: int, new_start_ts: datetime, confirmed: bool = False,
) -> dict[str, Any]:
    """Own appointments only. Autonomous outside the cancellation-fee window; inside it,
    needs the same in-conversation confirmation cancel_appointment requires."""
    args = {"appointment_id": appointment_id, "new_start_ts": new_start_ts, "confirmed": confirmed}
    with get_session() as session:
        appt = session.get(Appointment, appointment_id)
        if appt is None:
            write_audit(
                session, principal=principal, tool="reschedule_appointment", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        entity_ref = f"appointment:{appointment_id}"
        if principal.id is None or appt.customer_id != principal.id:
            reason = Reason.UNRESOLVED_PRINCIPAL.value if principal.id is None else Reason.PRINCIPAL_MISMATCH.value
            write_audit(
                session, principal=principal, tool="reschedule_appointment", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=reason, entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.DENIED.value, "reason": reason,
                "message": "We couldn't verify that appointment belongs to you. Please request a callback.",
            }

        if appt.status != "scheduled":
            write_audit(
                session, principal=principal, tool="reschedule_appointment", declared_tier=1,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value,
                     "message": "Only scheduled appointments can be rescheduled."}

        policy = load_policy(session)
        now = now_utc()
        item = session.get(ServiceItem, appt.service_item_id)
        new_end_ts = new_start_ts + timedelta(minutes=item.duration_min)

        fee_window_applies = cancellation_fee_applies(policy, appt.start_ts, now)
        if fee_window_applies and not confirmed:
            write_audit(
                session, principal=principal, tool="reschedule_appointment", declared_tier=1,
                decision=Decision.NEEDS_CONFIRM.value, args=args, reason=Reason.CANCELLATION_FEE.value,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.NEEDS_CONFIRM.value, "reason": Reason.CANCELLATION_FEE.value,
                "message": "Rescheduling this close to the appointment may incur a fee. Confirm to proceed.",
            }

        hours_ok, hours_reason = check_business_hours(policy, new_start_ts)
        lead_ok, lead_reason = check_lead_time(policy, new_start_ts, now)
        window_ok, window_reason = check_booking_window(policy, new_start_ts, now)
        tech = find_available_technician(session, item, new_start_ts, new_end_ts, exclude_appointment_id=appt.id)
        checks = [
            (hours_ok, hours_reason), (lead_ok, lead_reason),
            (window_ok, window_reason), (tech is not None, Reason.NO_SKILLED_TECH.value),
        ]
        first_fail = next((code for ok, code in checks if not ok), None)

        if first_fail is not None:
            preview = render_diff(
                entity_ref, before={"start_ts": str(appt.start_ts)}, after={"start_ts": str(new_start_ts)},
            )
            pending = queue_request(
                session, principal=principal, tool="reschedule_appointment", args=args, preview_text=preview,
            )
            write_audit(
                session, principal=principal, tool="reschedule_appointment", declared_tier=1,
                decision=Decision.QUEUED.value, args=args, reason=first_fail, entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.QUEUED.value, "request_id": pending.id,
                "reason": first_fail, "preview_text": pending.preview_text,
            }

        appt.start_ts = new_start_ts
        appt.end_ts = new_end_ts
        appt.technician_id = tech.id
        write_audit(
            session, principal=principal, tool="reschedule_appointment", declared_tier=1,
            decision=Decision.EXECUTED.value, args=args,
            reason=Reason.CANCELLATION_FEE.value if fee_window_applies else None,
            entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "appointment_id": appt.id, "start_ts": new_start_ts.isoformat()}


# --- Tier 2: writes requiring in-conversation confirmation -------------------------------


def cancel_appointment(
    *, principal: Principal, run_id: str | None = None, appointment_id: int, confirmed: bool = False
) -> dict[str, Any]:
    """Own appointments only. A cancellation fee inside the window must be disclosed and
    acknowledged before the appointment is actually cancelled."""
    args = {"appointment_id": appointment_id, "confirmed": confirmed}
    with get_session() as session:
        appt = session.get(Appointment, appointment_id)
        if appt is None:
            write_audit(
                session, principal=principal, tool="cancel_appointment", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}

        entity_ref = f"appointment:{appointment_id}"
        if principal.id is None or appt.customer_id != principal.id:
            reason = Reason.UNRESOLVED_PRINCIPAL.value if principal.id is None else Reason.PRINCIPAL_MISMATCH.value
            write_audit(
                session, principal=principal, tool="cancel_appointment", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=reason, entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.DENIED.value, "reason": reason,
                "message": "We couldn't verify that appointment belongs to you. Please request a callback.",
            }

        if appt.status != "scheduled":
            write_audit(
                session, principal=principal, tool="cancel_appointment", declared_tier=2,
                decision=Decision.DENIED.value, args=args, reason=Reason.INVALID_ARGUMENT.value,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value,
                     "message": "That appointment is already cancelled or completed."}

        policy = load_policy(session)
        now = now_utc()
        fee_applies = cancellation_fee_applies(policy, appt.start_ts, now)

        if fee_applies and not confirmed:
            write_audit(
                session, principal=principal, tool="cancel_appointment", declared_tier=2,
                decision=Decision.NEEDS_CONFIRM.value, args=args, reason=Reason.CANCELLATION_FEE.value,
                entity_ref=entity_ref, run_id=run_id,
            )
            session.commit()
            return {
                "decision": Decision.NEEDS_CONFIRM.value, "reason": Reason.CANCELLATION_FEE.value,
                "message": "Cancelling this close to the appointment may incur a fee. Confirm to proceed.",
            }

        appt.status = "cancelled"
        write_audit(
            session, principal=principal, tool="cancel_appointment", declared_tier=2,
            decision=Decision.EXECUTED.value, args=args,
            reason=Reason.CANCELLATION_FEE.value if fee_applies else None,
            entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {"decision": Decision.EXECUTED.value, "appointment_id": appt.id, "fee_applied": fee_applies}


# --- Tier 3: deferred approval -----------------------------------------------------------


def request_human_callback(*, principal: Principal, run_id: str | None = None, message: str) -> dict[str, Any]:
    """The universal escape hatch — always queues, no policy check gates it. This is the
    tool the agent should reach for on any ambiguity it can't (or shouldn't) resolve itself."""
    args = {"message": message}
    entity_ref = f"customer:{principal.id}" if principal.id is not None else None
    with get_session() as session:
        preview = render_diff("human callback request", before={}, after={"message": message})
        pending = queue_request(
            session, principal=principal, tool="request_human_callback", args=args, preview_text=preview,
        )
        write_audit(
            session, principal=principal, tool="request_human_callback", declared_tier=3,
            decision=Decision.QUEUED.value, args=args, entity_ref=entity_ref, run_id=run_id,
        )
        session.commit()
        return {
            "decision": Decision.QUEUED.value, "request_id": pending.id,
            "message": "A staff member will follow up with you.",
        }


REGISTRY_C: Registry = {
    "find_my_account": ToolSpec(fn=find_my_account_tool, tier=0),
    "list_services": ToolSpec(fn=list_services, tier=0),
    "get_availability": ToolSpec(fn=get_availability, tier=0),
    "get_my_appointments": ToolSpec(fn=get_my_appointments, tier=0),
    "get_quote": ToolSpec(fn=get_quote, tier=0),
    "get_payment_link": ToolSpec(fn=get_payment_link, tier=0),
    "book_appointment": ToolSpec(fn=book_appointment, tier=1),
    "reschedule_appointment": ToolSpec(fn=reschedule_appointment, tier=1),
    "cancel_appointment": ToolSpec(fn=cancel_appointment, tier=2),
    "request_human_callback": ToolSpec(fn=request_human_callback, tier=3),
}

assert len(REGISTRY_C) == 10, "Registry C must have exactly 10 tools"
