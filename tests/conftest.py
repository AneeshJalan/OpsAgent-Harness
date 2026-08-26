"""Shared fixtures for the tool-layer test suite. Every test gets its own throwaway SQLite
file so tests never share state or run order dependencies — `db.database`'s engine/session
factory is cached at module scope in production (one process, one DB), so tests reset those
globals themselves rather than importing a second copy of the module.
"""

from __future__ import annotations

import pytest

import db.database as database
from db.database import init_db


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    monkeypatch.setenv("OPS_DB_PATH", str(path))
    database._engine = None
    database._SessionLocal = None
    init_db(path)
    yield path
    database._engine = None
    database._SessionLocal = None


@pytest.fixture
def db_path_in_memory(monkeypatch):
    """Schema-only, against one shared in-memory SQLite connection instead of a real temp file
    -- for tests that exercise dispatch()'s own logic (registry membership, role gating,
    datetime coercion) without asserting on any DB row themselves. Real session/ORM behavior,
    just no file I/O; the engine is wired directly into db.database's module globals (rather
    than going through init_db against a second, separate engine) since SQLite's `:memory:`
    databases aren't shared across connections/engines the way a real file is."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from db.models import Base

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(database, "_engine", engine)
    monkeypatch.setattr(database, "_SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
    yield engine


@pytest.fixture
def edge_db(db_path):
    """Fresh schema plus only the hand-planted edge-case fixtures — fast, and what most tool
    tests actually need (known IDs, known mess). Use `full_db` instead for anything that
    needs bulk volume (e.g. exercising a scan across hundreds of rows)."""
    from db.seed_edge_cases import main as seed_edge_cases

    seed_edge_cases()
    return db_path


@pytest.fixture
def edge_db_with_policy(edge_db):
    """Edge cases plus policy_config, without the 150 rows of bulk filler — what most tool
    tests need: known fixture IDs, a real envelope to check requests against, fast to build."""
    from db.database import get_session
    from db.seed_bulk import seed_policy_config

    with get_session() as session:
        seed_policy_config(session)
        session.commit()
    return edge_db


@pytest.fixture
def full_db(edge_db):
    """Edge cases plus bulk filler and policy_config — the closest thing to the real DB."""
    from db.seed_bulk import main as seed_bulk

    seed_bulk()
    return edge_db


@pytest.fixture
def policy_only_db(db_path):
    """Fresh schema with just policy_config seeded — no customers/technicians/etc. Cheaper
    than edge_db for tests that only exercise the policy module."""
    from db.database import get_session
    from db.seed_bulk import seed_policy_config

    with get_session() as session:
        seed_policy_config(session)
        session.commit()
    return db_path
