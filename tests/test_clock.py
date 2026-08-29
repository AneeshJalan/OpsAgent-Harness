"""Keeps evals/clock.py's PATCH_TARGETS honest against the actual repo: a future module that
starts calling db.seed_common.now_utc() and forgets to be added to that list would otherwise
silently run against real wall-clock time during an eval run instead of the frozen reference --
this test fails loudly, immediately, the moment that drift happens."""

from __future__ import annotations

import re
from pathlib import Path

from evals.clock import PATCH_TARGETS

SRC_ROOT = Path(__file__).resolve().parents[1]

# Seed-generation scripts run once, at DB-build time, to produce the checked-in golden DB file --
# irrelevant to a live eval run's clock, so intentionally excluded from the patch surface.
_SEED_GENERATION_SCRIPTS = {"db/seed_edge_cases.py", "db/seed_bulk.py", "db/build_substitution_dbs.py"}

_NOW_UTC_IMPORT = re.compile(r"^from db\.seed_common import .*\bnow_utc\b", re.MULTILINE)


def _module_name_for(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT).as_posix()
    return rel.removesuffix(".py").replace("/", ".")


def _find_now_utc_consumers() -> set[str]:
    consumers = set()
    for path in SRC_ROOT.rglob("*.py"):
        rel = path.relative_to(SRC_ROOT).as_posix()
        if rel.startswith(("tests/", ".venv/")) or "__pycache__" in rel or rel in _SEED_GENERATION_SCRIPTS:
            continue
        if _NOW_UTC_IMPORT.search(path.read_text(encoding="utf-8")):
            consumers.add(_module_name_for(path))
    return consumers


def test_patch_targets_covers_every_now_utc_consumer_in_the_repo():
    found = _find_now_utc_consumers()
    # db.seed_common itself is in PATCH_TARGETS defensively (see clock.py's own docstring), not
    # because it's a `from ... import` consumer of itself -- exclude it from this side only.
    assert found == set(PATCH_TARGETS) - {"db.seed_common"}
