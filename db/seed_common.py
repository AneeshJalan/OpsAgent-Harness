"""Shared constants and helpers used by seed_edge_cases.py, seed_bulk.py, and
validate_seed.py — kept in one place so the three scripts can't quietly disagree on things
like "what counts as unpaid" or "where does the reserved ID block end."

EDGE_CASES.md is the human-readable version of the fixture list below; this module is the
source of truth both it and the validator are checked against.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from db.models import Customer, Invoice

# An invoice in either of these statuses still counts against the customer's balance.
# 'void' and 'paid' do not. Shared with validate_seed.py so the invariant check and the
# generators can never drift on the definition.
UNPAID_STATUSES = {"draft", "sent"}

FAKER_SEED = 42
BULK_CUSTOMER_COUNT = 150

# Fixed reference point bulk-generated appointments are scheduled around, so "past" and
# "future" stay stable regardless of what day this seeder actually runs on.
ANCHOR_DATE = datetime(2026, 8, 23)

# Hand-planted fixtures live below these IDs; bulk-generated rows start at these numbers.
# Bulk IDs are assigned explicitly (not left to autoincrement) so they don't shift if the
# planted fixture set ever grows — the golden set can reference low IDs and never chase a
# moving target.
CUSTOMER_BULK_START_ID = 201
APPOINTMENT_BULK_START_ID = 201
INVOICE_BULK_START_ID = 201
INVOICE_LINE_BULK_START_ID = 1001

# The planted fixture IDs, grouped by table — this is what validate_seed.py checks still
# resolve to live rows, and it's what EDGE_CASES.md describes in prose.
EDGE_CASE_CUSTOMER_IDS = list(range(1, 16))
EDGE_CASE_TECHNICIAN_IDS = list(range(1, 8))
EDGE_CASE_SERVICE_ITEM_IDS = list(range(1, 12))
EDGE_CASE_APPOINTMENT_IDS = list(range(1, 8))
EDGE_CASE_INVOICE_IDS = list(range(1, 4))

# Customer ID pairs the near-duplicate/hard-negative fixtures deliberately create. The
# DB-wide fuzzy scan in validate_seed.py checks name/address/phone similarity across every
# customer; without this whitelist it would re-discover (and flag as "unexpected") the exact
# pairs seed_edge_cases.py put there on purpose.
KNOWN_NAME_DUPLICATE_PAIRS = frozenset(
    {
        frozenset({1, 2}),  # Jonathan Reyes — formatting-only (phone/address punctuation)
        frozenset({3, 4}),  # Kathryn / Katheryn Munoz — typo
        frozenset({5, 6}),  # William / Bill Carter — semantic alias
        # Bonus fixture: the bulk generator incidentally drew "John Reyes" (customer 275),
        # a near-duplicate of the planted "Jonathan Reyes" pair (1, 2) — found by
        # validate_seed.py, not planted on purpose, and kept rather than suppressed because
        # it's exactly the kind of incidental collision bulk data is expected to produce.
        frozenset({1, 275}),
        frozenset({2, 275}),
    }
)
# Hard negatives: look like duplicates on one signal, are not the same person.
KNOWN_SAME_NAME_HARD_NEGATIVE_PAIRS = frozenset({frozenset({11, 12})})  # two "Maria Gonzalez"
KNOWN_SAME_ADDRESS_HARD_NEGATIVE_PAIRS = frozenset({frozenset({9, 10})})  # duplex, different units
KNOWN_SHARED_PHONE_PAIRS = frozenset({frozenset({7, 8})})  # household landline, two real people

# Appointment ID pairs known to overlap for the same technician — the planted one, plus
# bonus collisions the bulk generator produced incidentally (found by validate_seed.py, kept
# for the same reason as the customer near-duplicate bonus fixture above).
KNOWN_DOUBLE_BOOKED_APPOINTMENT_PAIRS = frozenset(
    {
        frozenset({1, 2}),  # planted: technician 7
        frozenset({215, 289}),  # bulk bonus: technician 1
        frozenset({202, 207}),  # bulk bonus: technician 4
    }
)


def now_utc() -> datetime:
    """Naive UTC datetime — the convention every timestamp column in this project follows."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def recompute_balances(session: Session) -> None:
    """Set every customer's balance_cents to the sum of their unpaid invoice totals.

    Both seeders call this once after they've created their invoices, instead of maintaining
    balance_cents by hand alongside every invoice insert — one place computes the derived
    value, so it can't drift from the invoices that are supposed to justify it.
    """
    totals: dict[int, int] = {}
    for (customer_id, total_cents) in session.query(Invoice.customer_id, Invoice.total_cents).filter(
        Invoice.status.in_(UNPAID_STATUSES)
    ):
        totals[customer_id] = totals.get(customer_id, 0) + total_cents

    for customer in session.query(Customer).all():
        customer.balance_cents = totals.get(customer.id, 0)
