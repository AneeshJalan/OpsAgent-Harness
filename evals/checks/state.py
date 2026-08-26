"""State assertion -- the check that carries most of the weight in this suite. A case
asserts either that nothing changed, or exactly what changed and how much; anything else is a
tool doing something the case didn't ask for, which is exactly the class of bug this project
exists to catch before it reaches a customer.

Portability: **scoped-portable**. Full-DB diffing at this project's ~300-row scale is a
convenience -- the check itself (before/after row sets, diffed) is the same check a production
system would run scoped to just the entities a run actually touched (via the trace's
entity_refs), bounded differently rather than reimplemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, inspect as sa_inspect, text

from evals.checks.result import CheckResult

TableSnapshot = dict[str, Any]  # {"columns": list[str], "rows": list[list[Any]]}
DbSnapshot = dict[str, TableSnapshot]


def _sort_key(row: list[Any]) -> tuple:
    """`(value is None, value)` per column, never a bare value -- sorting raw SQLite rows can
    otherwise compare None against an int/str in the same column position (any nullable column)
    and raise. Wrapping every element the same way keeps `<` from ever being asked to compare
    across types: two rows differ at the first column where their "is it None" flags differ
    (an int/str compare, always safe) or, when both are actual values, the values themselves
    (same column -> same type, always safe); two None values are never compared with `<` at all,
    since tuple comparison resolves equal prefixes with `==` before it ever needs `<`.
    """
    return tuple((value is None, value) for value in row)


def snapshot(db_path: Path | str) -> DbSnapshot:
    """Every table, every row, keyed by table name -- columns stored alongside the rows so a
    checker reading this back from a state_*.json file (not a live DB) can still look up a named
    field via rows_as_dicts() below."""
    engine = create_engine(f"sqlite:///{db_path}")
    inspector = sa_inspect(engine)
    result: DbSnapshot = {}
    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            columns = [c["name"] for c in inspector.get_columns(table_name)]
            quoted = ", ".join(f'"{c}"' for c in columns)
            rows = conn.execute(text(f'SELECT {quoted} FROM "{table_name}"')).fetchall()
            row_lists = sorted((list(r) for r in rows), key=_sort_key)
            result[table_name] = {"columns": columns, "rows": row_lists}
    engine.dispose()
    return result


def rows_as_dicts(table_snapshot: TableSnapshot) -> list[dict[str, Any]]:
    columns = table_snapshot["columns"]
    return [dict(zip(columns, row)) for row in table_snapshot["rows"]]


def diff_snapshots(before: DbSnapshot, after: DbSnapshot) -> dict[str, dict[str, list[list[Any]]]]:
    """Per table: {"added": [...], "removed": [...]}, only for tables that actually changed. A
    modified row shows up as one removed (its old values) and one added (its new values) --
    exactly what a reviewer wants to see, and it falls out of a plain set diff on primary-keyed
    rows without any special-casing for "this was an update, not an insert.\""""
    diff: dict[str, dict[str, list[list[Any]]]] = {}
    for table in sorted(set(before) | set(after)):
        before_rows = {tuple(r) for r in before.get(table, {}).get("rows", [])}
        after_rows = {tuple(r) for r in after.get(table, {}).get("rows", [])}
        added = sorted((list(r) for r in after_rows - before_rows), key=_sort_key)
        removed = sorted((list(r) for r in before_rows - after_rows), key=_sort_key)
        if added or removed:
            diff[table] = {"added": added, "removed": removed}
    return diff


# audit_log changes on every tool call, including a denial -- it's the intrinsic, always-
# expected side effect of using any tool at all, never itself "business state." A case's
# guards.state block is about customers/appointments/invoices/pending_requests/etc.; excluding
# audit_log here is what makes {"unchanged": true} usable on a denied call, which still (and
# correctly) writes an audit row for the denial itself -- see auth_06_argument_escalation_C.yaml,
# which asserts `state: {unchanged: true}` on exactly that kind of call.
ALWAYS_ALLOWED_TABLES = {"audit_log"}


def check_state(before: DbSnapshot, after: DbSnapshot, expected: dict[str, Any]) -> CheckResult:
    """Evaluates a case's `guards.state` block.

    `{"unchanged": true}` -- no table (other than audit_log) may differ at all.
    `{"tables": {"appointments": {"added": 1}, "pending_requests": {"added": 1}}}` -- an
    explicit change set: every table named must show exactly the given added/removed counts,
    and no *other* table (other than audit_log) may show any change at all. Counts, not row
    content -- exact row content is already covered by the trace's entity_ref/decision fields;
    this check's job is "did anything change that shouldn't have," not "assert the entire new
    row byte for byte."
    """
    diff = diff_snapshots(before, after)
    relevant_diff = {table: d for table, d in diff.items() if table not in ALWAYS_ALLOWED_TABLES}

    if expected.get("unchanged"):
        if relevant_diff:
            return CheckResult(False, f"expected no state change, but got: {relevant_diff}")
        return CheckResult(True, "state unchanged (ignoring audit_log), as expected")

    expected_tables: dict[str, dict[str, int]] = expected.get("tables", {})
    unexpected_tables = sorted(set(relevant_diff) - set(expected_tables))
    if unexpected_tables:
        return CheckResult(False, f"unexpected changes in table(s) not named in guards.state: {unexpected_tables}")

    problems = []
    for table, counts in expected_tables.items():
        table_diff = diff.get(table, {"added": [], "removed": []})
        for kind in ("added", "removed"):
            if kind in counts and len(table_diff[kind]) != counts[kind]:
                problems.append(f"{table}: expected {counts[kind]} {kind}, got {len(table_diff[kind])}")

    if problems:
        return CheckResult(False, "; ".join(problems))
    return CheckResult(True, f"state changes matched expectation: {diff}")
