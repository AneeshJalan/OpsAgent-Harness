"""Bulk filler data plus the business's real policy configuration.

Run second, after seed_edge_cases.py. Adds volume around the planted mess so dirty-data
detection has to work against something bigger than a handful of hand-picked rows, and seeds
`policy_config` — real business configuration, not planted mess, so it's an explicit step
here rather than something schema creation does implicitly.

Only customers/appointments/invoices/invoice_lines get bulk rows. Technicians and the service
catalog stay entirely hand-authored (seed_edge_cases.py) — a real business has a fixed roster
and a fixed price list, not thousands of Faker-generated job titles.

Deterministic: Faker.seed(FAKER_SEED) makes every generated value reproducible across
rebuilds. All randomness (which technician, which service, paid or not) is drawn from the
same seeded Faker instance rather than the stdlib `random` module, so one seed call
determines the entire run.
"""

from __future__ import annotations

from datetime import timedelta

from faker import Faker

from db.database import get_session
from db.models import Appointment, Customer, Invoice, InvoiceLine, PolicyConfig, ServiceItem, Technician
from db.seed_common import (
    ANCHOR_DATE,
    APPOINTMENT_BULK_START_ID,
    BULK_CUSTOMER_COUNT,
    CUSTOMER_BULK_START_ID,
    FAKER_SEED,
    INVOICE_BULK_START_ID,
    INVOICE_LINE_BULK_START_ID,
    now_utc,
    recompute_balances,
)

# The business's operating envelope — the knobs the policy engine reads at runtime. Stored
# as plain strings (simple, human-readable in the table), except business_hours, which is
# structured and stored as JSON so code can parse it without a bespoke mini-format.
POLICY_CONFIG_DEFAULTS = {
    "business_hours": '{"mon_fri": {"open": "08:00", "close": "18:00"}, '
    '"sat": {"open": "09:00", "close": "14:00"}, "sun": null}',
    "min_lead_time_hours": "4",
    "max_booking_window_days": "60",
    "auto_book_enabled": "true",
    "deposit_required_above": "50000",
    "blocking_balance_above": "25000",
    "max_discount_pct": "15",
    "cancellation_fee_window_hrs": "24",
    "after_hours_booking": "deferred",
    "identity_mode": "weak",
}

# Non-archived, non-null-price service items only — bulk history needs a real price to
# invoice against, and archived/null-price rows are the planted mess, not filler.
BULK_SERVICE_ITEM_POOL_IDS = [1, 2, 4, 5, 6, 7, 8, 9, 11]

# Appointments per customer: mostly one, some none, a few with two or three.
APPOINTMENT_COUNT_WEIGHTS = [(0, 30), (1, 45), (2, 18), (3, 7)]


def seed_policy_config(session) -> None:
    session.add_all(
        [PolicyConfig(key=key, value=value) for key, value in POLICY_CONFIG_DEFAULTS.items()]
    )


def _technicians_by_skill(session) -> dict[str, list[int]]:
    by_skill: dict[str, list[int]] = {}
    for tech in session.query(Technician).filter(Technician.active == 1):
        # skills_json is a JSON array string; avoid importing json just for this by treating
        # requires_skill match as substring containment, which is fine for our fixed,
        # comma-free skill vocabulary (plumbing, hvac, electrical, drain_cleaning).
        for skill in ("plumbing", "hvac", "electrical", "drain_cleaning"):
            if skill in tech.skills_json:
                by_skill.setdefault(skill, []).append(tech.id)
    return by_skill


def seed_bulk_customers_and_history(session, fake: Faker) -> None:
    service_items = {
        item.id: item
        for item in session.query(ServiceItem).filter(ServiceItem.id.in_(BULK_SERVICE_ITEM_POOL_IDS))
    }
    techs_by_skill = _technicians_by_skill(session)

    appointment_id = APPOINTMENT_BULK_START_ID
    invoice_id = INVOICE_BULK_START_ID
    invoice_line_id = INVOICE_LINE_BULK_START_ID

    for i in range(BULK_CUSTOMER_COUNT):
        customer_id = CUSTOMER_BULK_START_ID + i
        session.add(
            Customer(
                id=customer_id,
                name=fake.name(),
                phone=fake.phone_number(),
                email=fake.email(),
                address_line=fake.street_address(),
                city=fake.city(),
                zip=fake.postcode(),
                created_at=now_utc(),
            )
        )

        appointment_count = fake.random_element(
            elements=tuple(count for count, weight in APPOINTMENT_COUNT_WEIGHTS for _ in range(weight))
        )

        for _ in range(appointment_count):
            item_id = fake.random_element(elements=BULK_SERVICE_ITEM_POOL_IDS)
            item = service_items[item_id]
            candidate_techs = techs_by_skill.get(item.requires_skill, [])
            if not candidate_techs:
                continue
            technician_id = fake.random_element(elements=candidate_techs)

            # Skew toward the past (completed history) with a minority scheduled ahead.
            is_future = fake.boolean(chance_of_getting_true=20)
            if is_future:
                start = ANCHOR_DATE + timedelta(
                    days=fake.random_int(min=1, max=45), hours=fake.random_int(min=8, max=16)
                )
                status = "scheduled"
            else:
                start = ANCHOR_DATE - timedelta(
                    days=fake.random_int(min=1, max=180), hours=fake.random_int(min=8, max=16)
                )
                status = fake.random_element(elements=("completed",) * 9 + ("cancelled",))
            end = start + timedelta(minutes=item.duration_min)

            session.add(
                Appointment(
                    id=appointment_id,
                    customer_id=customer_id,
                    technician_id=technician_id,
                    service_item_id=item_id,
                    start_ts=start,
                    end_ts=end,
                    status=status,
                    created_by="seed",
                    created_via="seed",
                )
            )

            if status == "completed":
                is_paid = fake.boolean(chance_of_getting_true=70)
                issued_at = start + timedelta(hours=2)
                session.add(
                    Invoice(
                        id=invoice_id,
                        customer_id=customer_id,
                        appointment_id=appointment_id,
                        status="paid" if is_paid else "sent",
                        total_cents=item.base_price_cents,
                        issued_at=issued_at,
                        due_at=issued_at + timedelta(days=14),
                        paid_at=issued_at + timedelta(days=3) if is_paid else None,
                        processor_ref=f"ch_bulk_{invoice_id}" if is_paid else None,
                    )
                )
                session.add(
                    InvoiceLine(
                        id=invoice_line_id,
                        invoice_id=invoice_id,
                        service_item_id=item_id,
                        description=item.name,
                        qty=1,
                        unit_price_cents=item.base_price_cents,
                    )
                )
                invoice_id += 1
                invoice_line_id += 1

            appointment_id += 1


def main() -> None:
    Faker.seed(FAKER_SEED)
    fake = Faker()

    with get_session() as session:
        seed_policy_config(session)
        seed_bulk_customers_and_history(session, fake)
        recompute_balances(session)
        session.commit()
    print(f"Seeded {BULK_CUSTOMER_COUNT} bulk customers, policy_config, and their history.")


if __name__ == "__main__":
    main()
