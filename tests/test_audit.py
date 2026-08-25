"""audit.py: the write helper itself, and the provenance derivation it makes possible."""

from __future__ import annotations

from db.database import get_session
from db.models import AuditLog, Customer
from db.seed_common import now_utc
from tools.audit import is_provisional, write_audit
from tools.dispatcher import Decision
from tools.principal import Principal
from tools.reasons import Reason


def test_write_audit_inserts_expected_row(db_path):
    principal = Principal(type="customer", id=None)
    with get_session() as session:
        write_audit(
            session,
            principal=principal,
            tool="book_appointment",
            declared_tier=1,
            decision=Decision.EXECUTED.value,
            args={"service_item_id": 1},
            reason=Reason.UNRESOLVED_PRINCIPAL.value,
            entity_ref="customer:1",
            run_id="run-1",
        )
        session.commit()

    with get_session() as session:
        row = session.query(AuditLog).one()
        assert row.tool == "book_appointment"
        assert row.decision == Decision.EXECUTED.value
        assert row.reason == Reason.UNRESOLVED_PRINCIPAL.value
        assert row.entity_ref == "customer:1"
        assert row.principal_type == "customer"
        assert row.principal_id is None
        assert row.run_id == "run-1"
        assert row.outcome == "ok"
        assert '"service_item_id": 1' in row.args_json


def test_audit_and_state_write_share_one_transaction(db_path):
    """The convention this project depends on: a state write and its audit row commit
    together or not at all. Simulate the 'not at all' half by rolling back deliberately."""
    with get_session() as session:
        session.add(Customer(id=1, name="Test Customer", created_at=now_utc()))
        write_audit(
            session,
            principal=Principal(type="system", id=None),
            tool="book_appointment",
            declared_tier=1,
            decision=Decision.EXECUTED.value,
            args={},
            entity_ref="customer:1",
        )
        session.rollback()

    with get_session() as session:
        assert session.query(Customer).count() == 0
        assert session.query(AuditLog).count() == 0


def test_is_provisional_true_for_fall_forward_created_customer(db_path):
    with get_session() as session:
        session.add(Customer(id=1, name="Fallforward Customer", balance_cents=0, created_at=now_utc()))
        write_audit(
            session,
            principal=Principal(type="customer", id=None),
            tool="book_appointment",
            declared_tier=1,
            decision=Decision.EXECUTED.value,
            args={},
            reason=Reason.UNRESOLVED_PRINCIPAL.value,
            entity_ref="customer:1",
        )
        session.commit()

    with get_session() as session:
        assert is_provisional(session, 1) is True


def test_is_provisional_false_for_ordinary_customer(db_path):
    with get_session() as session:
        session.add(Customer(id=1, name="Ordinary Customer", balance_cents=0, created_at=now_utc()))
        session.commit()

    with get_session() as session:
        assert is_provisional(session, 1) is False


def test_is_provisional_false_once_merged(db_path):
    with get_session() as session:
        session.add(Customer(id=1, name="Survivor", balance_cents=0, created_at=now_utc()))
        session.add(Customer(id=2, name="Fallforward Customer", balance_cents=0, created_at=now_utc(), merged_into_id=1))
        write_audit(
            session,
            principal=Principal(type="customer", id=None),
            tool="book_appointment",
            declared_tier=1,
            decision=Decision.EXECUTED.value,
            args={},
            reason=Reason.AMBIGUOUS_IDENTITY.value,
            entity_ref="customer:2",
        )
        session.commit()

    with get_session() as session:
        assert is_provisional(session, 2) is False
