"""db/build_substitution_dbs.py: the three golden-DB copies the R14 (anti-oracle) checks replay
against must differ *only* in how many customers match PROBE_IDENTITY -- every other row, in
every other table, must be identical across all three files and identical to the golden DB they
were copied from. That invariant is the entire point of the script; these tests exist to prove
it holds, not just that the script runs.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.build_substitution_dbs import PROBE_IDENTITY, TARGETS, build_substitution_dbs, main
from db.models import (
    Appointment,
    AuditLog,
    Customer,
    Invoice,
    InvoiceLine,
    PendingRequest,
    PolicyConfig,
    ServiceItem,
    Technician,
)
from tools.identity import UNRESOLVED, find_my_account

# Every table build_substitution_dbs must leave untouched -- Customer is deliberately excluded,
# it's the one table these files are allowed to differ on.
UNTOUCHED_TABLES = [
    Technician, ServiceItem, Appointment, Invoice, InvoiceLine, PolicyConfig, PendingRequest, AuditLog,
]


def _session_for(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    return sessionmaker(bind=engine)()


def _snapshot(session, model) -> list[tuple]:
    columns = model.__table__.columns
    rows = session.query(model).all()
    return sorted(tuple(getattr(row, c.name) for c in columns) for row in rows)


def _exact_match_count(session, model=Customer) -> int:
    """How many customer rows carry the *exact* PROBE_IDENTITY tuple -- the literal thing
    TARGETS' counts describe. Deliberately not resolve_candidates(): that function only narrows
    when a field matches *someone*, so on a DB where nothing matches at all it hands back every
    customer, not zero -- correct for its own contract (find_my_account still reports
    UNRESOLVED either way), but the wrong tool for asserting how many rows exactly match."""
    return (
        session.query(model)
        .filter(
            model.name == PROBE_IDENTITY["name"],
            model.email == PROBE_IDENTITY["email"],
            model.phone == PROBE_IDENTITY["phone"],
            model.address_line == PROBE_IDENTITY["address"],
        )
        .count()
    )


def test_probe_identity_matches_nothing_in_the_golden_seed_data(full_db):
    """Sanity check on the constant itself: PROBE_IDENTITY must not collide with any real
    seeded customer, or 'zero matches' in ops_absent.db would be a lie. Checked two ways: no
    exact row match, and find_my_account (165 unrelated candidates, none of them a match on any
    field) still reports UNRESOLVED rather than accidentally resolving to a stranger."""
    with _session_for(full_db) as session:
        assert _exact_match_count(session) == 0
        assert find_my_account(session, **PROBE_IDENTITY) == UNRESOLVED


def test_missing_golden_db_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_substitution_dbs(golden_path=tmp_path / "does_not_exist.db", output_dir=tmp_path)


def test_builds_three_files_with_exactly_the_right_match_counts(full_db, tmp_path):
    built = build_substitution_dbs(golden_path=full_db, output_dir=tmp_path)
    assert set(built) == set(TARGETS)

    for filename, expected_count in TARGETS.items():
        with _session_for(built[filename]) as session:
            assert _exact_match_count(session) == expected_count, filename


def test_find_my_account_shape_invariance_holds_against_the_real_files(full_db, tmp_path):
    """The strongest form of the R14 check, run against the actual substitution files rather
    than hand-built in-process fixtures: absent and six-way-ambiguous must both return the
    UNRESOLVED sentinel: unresolved because a probe with someone else's info found no one, and
    unresolved because it found too many to safely pick one — the two failure modes must be
    indistinguishable from outside the function. Single must resolve to a real Principal."""
    built = build_substitution_dbs(golden_path=full_db, output_dir=tmp_path)

    with _session_for(built["ops_absent.db"]) as session:
        assert find_my_account(session, **PROBE_IDENTITY) == UNRESOLVED

    with _session_for(built["ops_six.db"]) as session:
        assert find_my_account(session, **PROBE_IDENTITY) == UNRESOLVED

    with _session_for(built["ops_single.db"]) as session:
        result = find_my_account(session, **PROBE_IDENTITY)
    assert result != UNRESOLVED
    assert result.type == "customer"
    assert result.id is not None


def test_every_non_customer_table_is_byte_identical_across_all_three_and_the_golden_db(full_db, tmp_path):
    built = build_substitution_dbs(golden_path=full_db, output_dir=tmp_path)

    with _session_for(full_db) as session:
        golden_snapshots = {model: _snapshot(session, model) for model in UNTOUCHED_TABLES}

    for filename in TARGETS:
        with _session_for(built[filename]) as session:
            for model in UNTOUCHED_TABLES:
                assert _snapshot(session, model) == golden_snapshots[model], (filename, model.__tablename__)


def test_customer_table_differs_only_by_the_appended_probe_rows(full_db, tmp_path):
    built = build_substitution_dbs(golden_path=full_db, output_dir=tmp_path)

    with _session_for(full_db) as session:
        golden_customers = _snapshot(session, Customer)
        golden_ids = {row.id for row in session.query(Customer.id).all()}

    for filename, expected_new_rows in TARGETS.items():
        with _session_for(built[filename]) as session:
            all_rows = session.query(Customer).all()
            existing = [r for r in all_rows if r.id in golden_ids]
            added = [r for r in all_rows if r.id not in golden_ids]

        assert sorted(
            tuple(getattr(r, c.name) for c in Customer.__table__.columns) for r in existing
        ) == golden_customers, filename
        assert len(added) == expected_new_rows, filename
        for row in added:
            assert row.name == PROBE_IDENTITY["name"]
            assert row.email == PROBE_IDENTITY["email"]
            assert row.phone == PROBE_IDENTITY["phone"]
            assert row.address_line == PROBE_IDENTITY["address"]


def test_main_builds_and_reports_all_three(full_db, tmp_path, monkeypatch, capsys):
    import db.build_substitution_dbs as build_mod

    monkeypatch.setattr(build_mod, "GOLDEN_DB_PATH", full_db)
    monkeypatch.setattr(build_mod, "OUTPUT_DIR", tmp_path)
    exit_code = main()
    out = capsys.readouterr().out
    assert exit_code == 0
    for filename in TARGETS:
        assert filename in out
        assert (tmp_path / filename).exists()
