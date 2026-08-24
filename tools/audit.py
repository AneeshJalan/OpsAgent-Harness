"""The in-transaction audit insert. One call, from inside the same Session as any state
write it describes, before that session's single commit() — never after a separate one.
Provenance (is this customer record provisional?) is derived from this table later, so a
state change that committed without its audit row would be a correctness bug, not a missing
log line.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from db.models import AuditLog
from db.seed_common import now_utc
from tools.principal import Principal


def write_audit(
    session: Session,
    *,
    principal: Principal,
    tool: str,
    declared_tier: int,
    decision: str,
    args: dict[str, Any],
    reason: str | None = None,
    entity_ref: str | None = None,
    outcome: str = "ok",
    run_id: str | None = None,
    idempotency_key: str | None = None,
) -> None:
    session.add(
        AuditLog(
            ts=now_utc(),
            run_id=run_id,
            principal_type=principal.type,
            principal_id=principal.id,
            principal_role=principal.role,
            tool=tool,
            declared_tier=declared_tier,
            decision=decision,
            reason=reason,
            args_json=json.dumps(args, default=str, sort_keys=True),
            entity_ref=entity_ref,
            outcome=outcome,
            idempotency_key=idempotency_key,
        )
    )


def is_provisional(session: Session, customer_id: int) -> bool:
    """A customer is provisional if book_appointment created it via fall-forward (logged
    reason ambiguous_identity or unresolved_principal against that entity_ref) and it hasn't
    since been soft-merged. Mirrors the query this project derives provenance from instead of
    a stored flag."""
    from db.models import Customer  # local import avoids a cycle with db.models at load time

    customer = session.get(Customer, customer_id)
    if customer is None or customer.merged_into_id is not None:
        return False

    hit = (
        session.query(AuditLog.id)
        .filter(
            AuditLog.tool == "book_appointment",
            AuditLog.entity_ref == f"customer:{customer_id}",
            AuditLog.reason.in_(["ambiguous_identity", "unresolved_principal"]),
        )
        .first()
    )
    return hit is not None
