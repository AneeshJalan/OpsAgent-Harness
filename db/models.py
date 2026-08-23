"""ORM models for ops.db.

A few things here are load-bearing elsewhere in the system and easy to "helpfully" change —
don't:

- Money columns are INTEGER cents everywhere. No Float/Numeric column exists in this file.
- No `provisional` column on `customers`, no `users` table, no `tenant_id`. Whether a customer
  record is provisional is derived from `audit_log` (see AuditLog below), not stored as a flag.
- FK constraints are declared (for documentation, joins, and relationships) but SQLite does
  not enforce them unless `PRAGMA foreign_keys=ON` is issued, which this project deliberately
  never does — a few seed fixtures are intentionally dangling references (an inactive
  technician still booked, an orphaned invoice line) and must insert and query cleanly rather
  than raising IntegrityError.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    phone: Mapped[str | None] = mapped_column(Text)
    email: Mapped[str | None] = mapped_column(Text)
    address_line: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(Text)
    zip: Mapped[str | None] = mapped_column(Text)
    # Derived from unpaid invoice totals — kept in sync by validation, not by a DB trigger.
    balance_cents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    internal_notes: Mapped[str | None] = mapped_column(Text)  # staff-visible only
    # Soft merge target — reversible, so eval cases need no DB rebuild between runs.
    merged_into_id: Mapped[int | None] = mapped_column(ForeignKey("customers.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class Technician(Base):
    __tablename__ = "technicians"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    skills_json: Mapped[str] = mapped_column(Text, nullable=False)
    home_zip: Mapped[str | None] = mapped_column(Text)
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class ServiceItem(Base):
    __tablename__ = "service_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    # NULL = quote-on-inspection; the get_quote tool must escalate rather than invent a price.
    base_price_cents: Mapped[int | None] = mapped_column(Integer)
    duration_min: Mapped[int] = mapped_column(Integer, nullable=False)
    requires_skill: Mapped[str | None] = mapped_column(Text)
    bookable_online: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    # Retired catalog entry — excluded from list_services/get_quote.
    archived: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Appointment(Base):
    __tablename__ = "appointments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    technician_id: Mapped[int | None] = mapped_column(ForeignKey("technicians.id"))
    service_item_id: Mapped[int] = mapped_column(ForeignKey("service_items.id"), nullable=False)
    start_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)  # scheduled | completed | cancelled
    created_by: Mapped[str | None] = mapped_column(Text)
    created_via: Mapped[str | None] = mapped_column(Text)  # agent_c | agent_s | seed | human
    idempotency_key: Mapped[str | None] = mapped_column(Text, unique=True)


class Invoice(Base):
    __tablename__ = "invoices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.id"), nullable=False)
    appointment_id: Mapped[int | None] = mapped_column(ForeignKey("appointments.id"))
    status: Mapped[str] = mapped_column(Text, nullable=False)  # draft | sent | paid | void
    total_cents: Mapped[int] = mapped_column(Integer, nullable=False)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime)
    due_at: Mapped[datetime | None] = mapped_column(DateTime)
    paid_at: Mapped[datetime | None] = mapped_column(DateTime)
    # Payment is reconciled, never asserted — record_payment takes a processor reference.
    processor_ref: Mapped[str | None] = mapped_column(Text)
    idempotency_key: Mapped[str | None] = mapped_column(Text, unique=True)


class InvoiceLine(Base):
    __tablename__ = "invoice_lines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    invoice_id: Mapped[int] = mapped_column(ForeignKey("invoices.id"), nullable=False)
    service_item_id: Mapped[int | None] = mapped_column(ForeignKey("service_items.id"))
    description: Mapped[str | None] = mapped_column(Text)
    qty: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    unit_price_cents: Mapped[int] = mapped_column(Integer, nullable=False)


class PolicyConfig(Base):
    """The business's operating envelope (business hours, lead time, discount cap, etc.).
    Seeded explicitly by the bulk seeder — this is real configuration, not planted mess, so
    schema creation leaves this table empty."""

    __tablename__ = "policy_config"

    key: Mapped[str] = mapped_column(Text, primary_key=True)
    value: Mapped[str] = mapped_column(Text, nullable=False)


class PendingRequest(Base):
    """The deferred-approval queue. A row here means 'not done yet' — a separate approve
    step is the only thing that executes the underlying tool call, standing in for a human
    decision."""

    __tablename__ = "pending_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    requested_by_type: Mapped[str] = mapped_column(Text, nullable=False)
    requested_by_id: Mapped[int | None] = mapped_column(Integer)
    tool: Mapped[str] = mapped_column(Text, nullable=False)
    args_json: Mapped[str] = mapped_column(Text, nullable=False)
    preview_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime)
    resolved_by: Mapped[str | None] = mapped_column(Text)


class AuditLog(Base):
    """The system of record for both provenance and observability. Every write-path tool
    inserts here in the SAME transaction as the state change it describes (see database.py) —
    `reason` and `entity_ref` are load-bearing for correctness (e.g. deriving whether a
    customer record is provisional), not just logging detail."""

    __tablename__ = "audit_log"
    __table_args__ = (
        Index("idx_audit_entity", "tool", "entity_ref"),
        Index("idx_audit_run", "run_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    run_id: Mapped[str | None] = mapped_column(Text)
    principal_type: Mapped[str] = mapped_column(Text, nullable=False)  # customer | staff | system
    principal_id: Mapped[int | None] = mapped_column(Integer)  # NULL = unresolved
    principal_role: Mapped[str | None] = mapped_column(Text)  # dispatcher | manager | owner
    tool: Mapped[str] = mapped_column(Text, nullable=False)
    declared_tier: Mapped[int] = mapped_column(Integer, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)  # executed|needs_confirm|queued|denied
    reason: Mapped[str | None] = mapped_column(Text)  # reason code, e.g. 'lead_time', 'balance_hold'
    args_json: Mapped[str] = mapped_column(Text, nullable=False)
    entity_ref: Mapped[str | None] = mapped_column(Text)  # e.g. 'customer:412'
    outcome: Mapped[str] = mapped_column(Text, nullable=False)  # ok | error
    idempotency_key: Mapped[str | None] = mapped_column(Text)
