"""Post-seed invariant checker. Run after seed_edge_cases.py and seed_bulk.py; exits non-zero
if anything below fails, so a broken seed gets caught before it silently corrupts golden-set
assumptions.

Bulk Faker data will incidentally produce its own near-duplicate names and technician
double-bookings at any real volume — that's expected, not a bug in the generator. This script
flags every such collision that isn't already accounted for in seed_common.py's known-pairs
lists, so each one gets triaged deliberately: either tighten the bulk generator to avoid it,
or add it to the known-pairs list (and to EDGE_CASES.md) as a documented bonus fixture. What
it must never do is ship silently.
"""

from __future__ import annotations

import sys
from difflib import SequenceMatcher
from itertools import combinations

from db.database import get_session
from db.models import Appointment, Customer
from db.seed_common import (
    EDGE_CASE_APPOINTMENT_IDS,
    EDGE_CASE_CUSTOMER_IDS,
    EDGE_CASE_INVOICE_IDS,
    EDGE_CASE_SERVICE_ITEM_IDS,
    EDGE_CASE_TECHNICIAN_IDS,
    KNOWN_DOUBLE_BOOKED_APPOINTMENT_PAIRS,
    KNOWN_NAME_DUPLICATE_PAIRS,
    KNOWN_SAME_ADDRESS_HARD_NEGATIVE_PAIRS,
    KNOWN_SAME_NAME_HARD_NEGATIVE_PAIRS,
    KNOWN_SHARED_PHONE_PAIRS,
    UNPAID_STATUSES,
)

NAME_SIMILARITY_THRESHOLD = 0.82

# Pairs already accounted for by a planted fixture — never worth re-reporting.
_ALL_KNOWN_PAIRS = (
    KNOWN_NAME_DUPLICATE_PAIRS
    | KNOWN_SAME_NAME_HARD_NEGATIVE_PAIRS
    | KNOWN_SAME_ADDRESS_HARD_NEGATIVE_PAIRS
    | KNOWN_SHARED_PHONE_PAIRS
)


def _normalize(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def check_balance_invariant(session) -> list[str]:
    errors = []
    invoices_by_customer: dict[int, int] = {}
    from db.models import Invoice  # local import keeps the top-level import list focused

    for customer_id, total_cents in session.query(Invoice.customer_id, Invoice.total_cents).filter(
        Invoice.status.in_(UNPAID_STATUSES)
    ):
        invoices_by_customer[customer_id] = invoices_by_customer.get(customer_id, 0) + total_cents

    for customer in session.query(Customer).all():
        expected = invoices_by_customer.get(customer.id, 0)
        if customer.balance_cents != expected:
            errors.append(
                f"customer {customer.id}: balance_cents={customer.balance_cents} but "
                f"unpaid invoice total is {expected}"
            )
    return errors


def check_fixture_ids_exist(session) -> list[str]:
    errors = []
    checks = [
        (Customer, EDGE_CASE_CUSTOMER_IDS, "customers"),
        (Appointment, EDGE_CASE_APPOINTMENT_IDS, "appointments"),
    ]
    from db.models import Invoice, ServiceItem, Technician

    checks += [
        (Technician, EDGE_CASE_TECHNICIAN_IDS, "technicians"),
        (ServiceItem, EDGE_CASE_SERVICE_ITEM_IDS, "service_items"),
        (Invoice, EDGE_CASE_INVOICE_IDS, "invoices"),
    ]
    for model, ids, label in checks:
        found_ids = {row.id for row in session.query(model.id).filter(model.id.in_(ids))}
        missing = sorted(set(ids) - found_ids)
        if missing:
            errors.append(f"{label}: missing planted fixture ids {missing}")
    return errors


def check_customer_similarity(session) -> tuple[list[str], list[str]]:
    """Returns (failures, informational_notes). A failure is a collision outside the
    known-pairs whitelist; an informational note documents a known pair was found and
    confirms it as the fixture it's supposed to be (not a failure)."""
    customers = session.query(Customer).all()
    failures: list[str] = []
    notes: list[str] = []

    for a, b in combinations(customers, 2):
        pair = frozenset({a.id, b.id})
        is_known = pair in _ALL_KNOWN_PAIRS

        same_phone = a.phone and b.phone and a.phone == b.phone
        name_ratio = SequenceMatcher(None, _normalize(a.name), _normalize(b.name)).ratio()
        same_address = (
            _normalize(a.address_line) == _normalize(b.address_line) != ""
            and _normalize(a.city) == _normalize(b.city)
            and _normalize(a.zip) == _normalize(b.zip)
        )

        signal = None
        if same_phone:
            signal = "shared phone"
        elif name_ratio >= NAME_SIMILARITY_THRESHOLD:
            signal = f"similar name (ratio={name_ratio:.2f})"
        elif same_address:
            signal = "same address"

        if signal is None:
            continue
        if is_known:
            notes.append(f"customers {a.id}/{b.id}: {signal} — known fixture, OK")
        else:
            failures.append(
                f"customers {a.id}/{b.id} ('{a.name}' / '{b.name}'): {signal} — "
                f"not in any known-pairs list in seed_common.py"
            )
    return failures, notes


def check_technician_double_bookings(session) -> tuple[list[str], list[str]]:
    appointments = (
        session.query(Appointment)
        .filter(Appointment.technician_id.isnot(None), Appointment.status != "cancelled")
        .order_by(Appointment.technician_id, Appointment.start_ts)
        .all()
    )
    by_tech: dict[int, list[Appointment]] = {}
    for appt in appointments:
        by_tech.setdefault(appt.technician_id, []).append(appt)

    failures: list[str] = []
    notes: list[str] = []
    for tech_id, appts in by_tech.items():
        for a, b in combinations(appts, 2):
            if a.start_ts < b.end_ts and b.start_ts < a.end_ts:
                msg = (
                    f"technician {tech_id}: appointments {a.id} "
                    f"({a.start_ts}-{a.end_ts}) and {b.id} ({b.start_ts}-{b.end_ts}) overlap"
                )
                if frozenset({a.id, b.id}) in KNOWN_DOUBLE_BOOKED_APPOINTMENT_PAIRS:
                    notes.append(msg + " — known fixture, OK")
                else:
                    failures.append(msg)
    return failures, notes


def main() -> int:
    ok = True
    with get_session() as session:
        checks = [
            ("balance invariant", check_balance_invariant(session)),
            ("fixture id resolution", check_fixture_ids_exist(session)),
        ]

        sim_failures, sim_notes = check_customer_similarity(session)
        booking_failures, booking_notes = check_technician_double_bookings(session)
        checks.append(("customer similarity scan", sim_failures))
        checks.append(("technician double-booking scan", booking_failures))

        for label, notes in (("customer similarity scan", sim_notes), ("technician double-booking scan", booking_notes)):
            for note in notes:
                print(f"[info] {label}: {note}")

        for label, errors in checks:
            if errors:
                ok = False
                print(f"[FAIL] {label}:")
                for e in errors:
                    print(f"  - {e}")
            else:
                print(f"[ok] {label}")

    if not ok:
        print("\nvalidate_seed.py: FAILED — see above. Triage each [FAIL] line: either tighten "
              "the bulk generator, or document it in seed_common.py's known-pairs lists and "
              "EDGE_CASES.md as an accepted bonus fixture.")
        return 1
    print("\nvalidate_seed.py: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
