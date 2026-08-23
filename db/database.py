"""Engine + session factory for ops.db.

One `Session` per tool call, one `commit()` at the end of that call covering both the state
write and its audit_log row — never autocommit-per-statement, and never two separate commits
for the two writes. Provenance (is a customer record provisional?) gets derived from
audit_log later, so a state change that committed without its audit row is a correctness bug,
not just a missing log line.

Usage inside a tool function:

    from db.database import get_session

    with get_session() as session:
        ... state writes ...
        ... audit_log insert ...
        session.commit()   # both together, or neither (exception -> context manager rolls back)
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

DEFAULT_DB_PATH = Path("db/opsagent.db")


def _resolve_db_path() -> Path:
    override = os.environ.get("OPS_DB_PATH")
    return Path(override) if override else DEFAULT_DB_PATH


def get_engine(db_path: Path | None = None) -> Engine:
    """Create the SQLAlchemy engine. No FK-enforcement pragma is attached — SQLite does not
    enforce declared ForeignKeys unless PRAGMA foreign_keys=ON is issued, and this project
    deliberately never issues it: several seed fixtures are intentionally dangling references
    (an inactive technician still booked, an orphaned invoice line) and must insert and query
    cleanly rather than raising IntegrityError."""
    path = db_path if db_path is not None else _resolve_db_path()
    return create_engine(f"sqlite:///{path}")


# Module-level default engine/session factory, built lazily so importing this module never
# touches disk by itself (tests and scripts can pass their own db_path to get_engine/init_db).
_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def _default_engine() -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        _engine = get_engine()
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    """Return a new Session bound to the default engine. Caller is responsible for one
    commit() (or letting an exception roll back) and for closing/using it as a context manager."""
    _default_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def init_db(db_path: Path | None = None, *, drop_first: bool = False) -> Engine:
    """Create all tables (schema only — no rows)."""
    engine = get_engine(db_path)
    if drop_first:
        Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    return engine


if __name__ == "__main__":
    eng = init_db()
    print(f"Created tables at {eng.url}")
