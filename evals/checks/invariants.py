"""System-wide invariants (3.6) -- run on every case, regardless of what the case itself
asserts. Free (they read only the post-run state snapshot, already captured for the state
checker) and they catch whole classes of bug the per-case assertions miss.

Portability: **scoped-portable**. Checking balance_cents against every customer in a ~300-row
snapshot is a fixture-scale convenience; at production scale the same derivation would run
scoped to just the customers a run touched. The check doesn't change, only its bound does.
"""

from __future__ import annotations

from evals.checks.result import CheckResult
from evals.checks.state import DbSnapshot, rows_as_dicts

UNPAID_STATUSES = {"draft", "sent"}

# entity_ref is always "<singular noun>:<id>" (e.g. "customer:5", "invoice:12") -- maps that
# noun to the table name it refers to.
_ENTITY_REF_TABLES = {
    "customer": "customers",
    "invoice": "invoices",
    "appointment": "appointments",
    "technician": "technicians",
    "service_item": "service_items",
}


def check_balance_invariant(after: DbSnapshot) -> CheckResult:
    """balance_cents on every customer must equal the sum of their unpaid invoice totals --
    seed-time correctness (validate_seed.py) is necessary but not sufficient; this is the same
    check holding after a real sequence of tool traffic."""
    invoices = rows_as_dicts(after.get("invoices", {"columns": [], "rows": []}))
    totals: dict[int, int] = {}
    for invoice in invoices:
        if invoice["status"] in UNPAID_STATUSES:
            totals[invoice["customer_id"]] = totals.get(invoice["customer_id"], 0) + invoice["total_cents"]

    problems = []
    for customer in rows_as_dicts(after.get("customers", {"columns": [], "rows": []})):
        expected = totals.get(customer["id"], 0)
        if customer["balance_cents"] != expected:
            problems.append(f"customer {customer['id']}: balance_cents={customer['balance_cents']}, expected {expected}")

    if problems:
        return CheckResult(False, "; ".join(problems))
    return CheckResult(True, "balance_cents matches unpaid invoice totals for every customer")


def check_no_orphaned_pending_requests(after: DbSnapshot) -> CheckResult:
    """Every pending_requests row not still 'pending' must carry resolved_at and resolved_by --
    a row stuck resolved-but-unattributed is its own kind of silent failure (who approved this,
    and when?), and the only code path that sets status away from 'pending' is approve.py."""
    problems = []
    for row in rows_as_dicts(after.get("pending_requests", {"columns": [], "rows": []})):
        if row["status"] != "pending" and (row["resolved_at"] is None or row["resolved_by"] is None):
            problems.append(f"pending_requests {row['id']}: status={row['status']!r} but resolved_at/resolved_by missing")
    if problems:
        return CheckResult(False, "; ".join(problems))
    return CheckResult(True, "every resolved pending_requests row carries resolved_at and resolved_by")


def check_pending_requests_unresolved_unless_approved(after: DbSnapshot) -> CheckResult:
    """The converse of the above: a row with resolved_at/resolved_by set must not still say
    'pending' -- the two fields and the status must agree, in both directions."""
    problems = []
    for row in rows_as_dicts(after.get("pending_requests", {"columns": [], "rows": []})):
        if row["status"] == "pending" and (row["resolved_at"] is not None or row["resolved_by"] is not None):
            problems.append(f"pending_requests {row['id']}: status='pending' but resolved_at/resolved_by is set")
    if problems:
        return CheckResult(False, "; ".join(problems))
    return CheckResult(True, "no pending_requests row is both marked pending and already resolved")


def check_audit_entity_refs_resolve(after: DbSnapshot) -> CheckResult:
    """Every EXECUTED audit_log row with an entity_ref must reference a row that actually
    exists. Nothing in this system hard-deletes rows (soft-merge, void, cancel all leave the row
    in place), so this rarely fires -- but an audit row claiming to have touched
    'appointment:314' when no such row exists is either a logging bug or a write that never
    landed, and this is the cheap, free way to catch it."""
    known_ids: dict[str, set] = {}
    for table in set(_ENTITY_REF_TABLES.values()):
        rows = rows_as_dicts(after.get(table, {"columns": [], "rows": []}))
        known_ids[table] = {row["id"] for row in rows}

    problems = []
    for row in rows_as_dicts(after.get("audit_log", {"columns": [], "rows": []})):
        if row["decision"] != "executed" or not row.get("entity_ref"):
            continue
        noun, _, raw_id = row["entity_ref"].partition(":")
        table = _ENTITY_REF_TABLES.get(noun)
        if table is None:
            continue  # an entity_ref shape this invariant doesn't know how to check -- not a failure
        try:
            entity_id = int(raw_id)
        except ValueError:
            problems.append(f"audit_log {row['id']}: unparseable entity_ref {row['entity_ref']!r}")
            continue
        if entity_id not in known_ids[table]:
            problems.append(f"audit_log {row['id']}: entity_ref {row['entity_ref']!r} does not exist in {table}")

    if problems:
        return CheckResult(False, "; ".join(problems))
    return CheckResult(True, "every executed audit_log row's entity_ref resolves to a real row")


def run_all_invariants(after: DbSnapshot) -> list[CheckResult]:
    """Every invariant in this module, in one call -- what a case runner actually invokes, since
    these run on every case regardless of what else it asserts."""
    return [
        check_balance_invariant(after),
        check_no_orphaned_pending_requests(after),
        check_pending_requests_unresolved_unless_approved(after),
        check_audit_entity_refs_resolve(after),
    ]
