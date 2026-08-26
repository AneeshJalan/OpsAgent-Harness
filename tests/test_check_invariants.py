"""evals/checks/invariants.py: system-wide invariants, run on every case. Each check is proven
two ways -- it passes on a real, healthy post-run snapshot, and it actually catches the specific
bug it claims to catch when that bug is deliberately injected. A checker that only ever passes
proves nothing about whether it can fail.
"""

from __future__ import annotations

import copy

import approve
from db.database import get_session
from db.models import Customer, PendingRequest
from evals.checks.invariants import (
    check_audit_entity_refs_resolve,
    check_balance_invariant,
    check_no_orphaned_pending_requests,
    check_pending_requests_unresolved_unless_approved,
    run_all_invariants,
)
from evals.checks.state import snapshot
from tools.dispatcher import dispatch
from tools.principal import Principal
from tools.registry_s import REGISTRY_S

DISPATCHER = Principal(type="staff", id=1, role="dispatcher")
MANAGER = Principal(type="staff", id=2, role="manager")


def test_all_invariants_pass_on_a_healthy_post_seed_snapshot(edge_db_with_policy):
    after = snapshot(edge_db_with_policy)
    results = run_all_invariants(after)
    failures = [r.detail for r in results if not r.passed]
    assert failures == []


def test_all_invariants_pass_after_a_real_queue_and_approve_sequence(edge_db_with_policy):
    write_off = dispatch(REGISTRY_S, "write_off_balance", DISPATCHER, customer_id=13, note="bad debt")
    approve.approve(write_off["request_id"], approver_role="manager", approver_id=2)

    after = snapshot(edge_db_with_policy)
    failures = [r.detail for r in run_all_invariants(after) if not r.passed]
    assert failures == []


def test_balance_invariant_catches_a_tampered_balance(edge_db_with_policy):
    with get_session() as session:
        customer = session.query(Customer).first()
        customer.balance_cents += 999
        session.commit()

    after = snapshot(edge_db_with_policy)
    result = check_balance_invariant(after)
    assert result.passed is False
    assert str(customer.id) in result.detail


def test_orphaned_pending_request_is_caught(edge_db_with_policy):
    dispatch(REGISTRY_S, "void_invoice", DISPATCHER, invoice_id=3)
    after = snapshot(edge_db_with_policy)
    # tamper: force the row's status to look resolved without setting resolved_at/resolved_by
    tampered = copy.deepcopy(after)
    cols = tampered["pending_requests"]["columns"]
    for row in tampered["pending_requests"]["rows"]:
        row[cols.index("status")] = "executed"

    result = check_no_orphaned_pending_requests(tampered)
    assert result.passed is False


def test_pending_request_marked_resolved_but_still_pending_status_is_caught(edge_db_with_policy):
    dispatch(REGISTRY_S, "void_invoice", DISPATCHER, invoice_id=3)
    after = snapshot(edge_db_with_policy)
    tampered = copy.deepcopy(after)
    cols = tampered["pending_requests"]["columns"]
    resolved_at_idx = cols.index("resolved_at")
    resolved_by_idx = cols.index("resolved_by")
    for row in tampered["pending_requests"]["rows"]:
        row[resolved_at_idx] = "2026-01-01T00:00:00"
        row[resolved_by_idx] = "manager:2"
        # status left as "pending" -- contradicts the now-set resolution fields

    result = check_pending_requests_unresolved_unless_approved(tampered)
    assert result.passed is False


def test_audit_entity_ref_pointing_at_a_nonexistent_row_is_caught(edge_db_with_policy):
    after = snapshot(edge_db_with_policy)
    tampered = copy.deepcopy(after)
    cols = tampered["audit_log"]["columns"]
    decision_idx, entity_ref_idx = cols.index("decision"), cols.index("entity_ref")
    # inject a fabricated executed row pointing at a customer id that cannot exist
    template = list(tampered["audit_log"]["rows"][0]) if tampered["audit_log"]["rows"] else [None] * len(cols)
    fabricated = list(template)
    fabricated[decision_idx] = "executed"
    fabricated[entity_ref_idx] = "customer:999999"
    if "id" in cols:
        fabricated[cols.index("id")] = -1
    tampered["audit_log"]["rows"].append(fabricated)

    result = check_audit_entity_refs_resolve(tampered)
    assert result.passed is False
    assert "999999" in result.detail
