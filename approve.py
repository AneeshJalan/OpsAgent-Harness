"""Simulates the human half of the Tier-3 approval flow. Not a UI — a CLI stand-in for "a
manager looked at the queue and made a call." Nothing in this codebase calls this script
automatically; a QUEUED request stays queued (and its underlying rows untouched) until a human
(or a test, standing in for one) runs this by hand.

Usage:
    python approve.py <request_id> --approve --role manager [--approver-id 7]
    python approve.py <request_id> --reject  --role dispatcher

Registry S's Tier-3 tools (write_off_balance, void_invoice, merge_customers) and an
over-cap apply_discount require a manager or owner approver — anyone lower is refused here,
before anything is touched. Registry C's escalated requests (book_appointment,
reschedule_appointment, request_human_callback) accept any staff role, matching the spec's
"Staff (for C)" approver rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

from db.database import get_session
from db.models import Invoice, PendingRequest
from db.seed_common import now_utc
from tools.audit import write_audit
from tools.dispatcher import Decision
from tools.principal import Principal
from tools.reasons import Reason
from tools.registry_c import _execute_queued_booking, _execute_queued_reschedule
from tools.registry_s import (
    _execute_apply_discount,
    _execute_merge_customers,
    _execute_void_invoice,
    _execute_write_off_balance,
)

MANAGER_GATED_TOOLS = {"write_off_balance", "void_invoice", "merge_customers", "apply_discount"}

DECLARED_TIER = {
    "book_appointment": 1, "reschedule_appointment": 1, "apply_discount": 2,
    "write_off_balance": 3, "void_invoice": 3, "merge_customers": 3, "request_human_callback": 3,
}


def _apply_discount_executor(session, args: dict[str, Any]) -> dict[str, Any]:
    invoice = session.get(Invoice, args["invoice_id"])
    if invoice is None:
        return {"decision": Decision.DENIED.value, "reason": Reason.INVALID_ARGUMENT.value}
    result = _execute_apply_discount(session, invoice, args["discount_pct"])
    return {"decision": Decision.EXECUTED.value, "entity_ref": f"invoice:{invoice.id}", **result}


def _write_off_balance_executor(session, args: dict[str, Any]) -> dict[str, Any]:
    result = _execute_write_off_balance(session, args["customer_id"])
    return {"decision": Decision.EXECUTED.value, "entity_ref": f"customer:{args['customer_id']}", **result}


def _void_invoice_executor(session, args: dict[str, Any]) -> dict[str, Any]:
    result = _execute_void_invoice(session, args["invoice_id"])
    return {"decision": Decision.EXECUTED.value, "entity_ref": f"invoice:{args['invoice_id']}", **result}


def _merge_customers_executor(session, args: dict[str, Any]) -> dict[str, Any]:
    result = _execute_merge_customers(session, survivor_id=args["survivor_id"], loser_id=args["loser_id"])
    return {"decision": Decision.EXECUTED.value, "entity_ref": f"customer:{args['loser_id']}", **result}


def _callback_executor(session, args: dict[str, Any]) -> dict[str, Any]:
    """There's no further state to change — the request itself *was* the deliverable. This
    executor just lets request_human_callback flow through the same approve/reject path as
    every other Tier-3 tool."""
    return {"decision": Decision.EXECUTED.value, "message": "Callback completed."}


EXECUTORS: dict[str, Callable[[Any, dict[str, Any]], dict[str, Any]]] = {
    "book_appointment": lambda session, args: _execute_queued_booking(session, args),
    "reschedule_appointment": lambda session, args: _execute_queued_reschedule(session, args),
    "apply_discount": _apply_discount_executor,
    "write_off_balance": _write_off_balance_executor,
    "void_invoice": _void_invoice_executor,
    "merge_customers": _merge_customers_executor,
    "request_human_callback": _callback_executor,
}


def approve(request_id: int, *, approver_role: str, approver_id: int, run_id: str | None = None) -> dict[str, Any]:
    with get_session() as session:
        pending = session.get(PendingRequest, request_id)
        if pending is None:
            return {"error": f"No pending_requests row with id {request_id}."}
        if pending.status != "pending":
            return {"error": f"Request {request_id} is already '{pending.status}', not pending."}
        if pending.tool in MANAGER_GATED_TOOLS and approver_role not in ("manager", "owner"):
            return {"error": f"'{pending.tool}' requires a manager or owner approver (got '{approver_role}')."}

        executor = EXECUTORS.get(pending.tool)
        if executor is None:
            return {"error": f"No approval executor registered for tool '{pending.tool}'."}

        args = json.loads(pending.args_json)
        result = executor(session, args)

        principal = Principal(type="staff", id=approver_id, role=approver_role)
        write_audit(
            session, principal=principal, tool=pending.tool,
            declared_tier=DECLARED_TIER.get(pending.tool, 3),
            decision=result.get("decision", Decision.EXECUTED.value),
            args=args, reason=result.get("reason"), entity_ref=result.get("entity_ref"), run_id=run_id,
        )

        pending.status = "rejected" if result.get("decision") == Decision.DENIED.value else "executed"
        pending.resolved_at = now_utc()
        pending.resolved_by = f"{approver_role}:{approver_id}"
        session.commit()

        return {"request_id": request_id, "tool": pending.tool, "status": pending.status, "result": result}


def reject(request_id: int, *, approver_role: str, approver_id: int) -> dict[str, Any]:
    with get_session() as session:
        pending = session.get(PendingRequest, request_id)
        if pending is None:
            return {"error": f"No pending_requests row with id {request_id}."}
        if pending.status != "pending":
            return {"error": f"Request {request_id} is already '{pending.status}', not pending."}

        pending.status = "rejected"
        pending.resolved_at = now_utc()
        pending.resolved_by = f"{approver_role}:{approver_id}"
        session.commit()
        return {"request_id": request_id, "tool": pending.tool, "status": "rejected"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve a Tier-3 pending_requests row.")
    parser.add_argument("request_id", type=int)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--approve", action="store_true")
    group.add_argument("--reject", action="store_true")
    parser.add_argument("--role", required=True, choices=["dispatcher", "manager", "owner"],
                         help="The approving staff member's role.")
    parser.add_argument("--approver-id", type=int, default=0)
    args = parser.parse_args(argv)

    if args.approve:
        result = approve(args.request_id, approver_role=args.role, approver_id=args.approver_id)
    else:
        result = reject(args.request_id, approver_role=args.role, approver_id=args.approver_id)

    print(json.dumps(result, indent=2, default=str))
    return 1 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
