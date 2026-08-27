"""Structural validation for the golden eval case corpus (evals/cases/) -- not the checkers
themselves (those are tested separately), just that every case file is well-formed, uniquely
identified, and references real fixtures. A case file with a typo'd id or a principal pointing
at a customer that doesn't exist is a case bug that would otherwise only surface as a confusing
failure deep into a real (expensive) run.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest
import yaml

from db.database import get_session
from db.models import Customer

CASES_DIR = Path(__file__).resolve().parent.parent / "evals" / "cases"
REQUIRED_TOP_LEVEL_KEYS = {"id", "category", "persona", "risks", "db", "principal", "turns", "guards", "scored"}

# The planned category distribution -- exact category -> count this corpus must match.
EXPECTED_DISTRIBUTION = {
    "happy_path": 8,
    "ambiguity": 6,
    "identity_scoping": 6,
    "authorization": 10,
    "policy": 6,
    "dirty_data": 5,
    "hallucination": 4,
    "over_escalation": 3,
    "provisional": 2,
}


def _case_files() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(CASES_DIR / "**" / "*.yaml"), recursive=True))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


CASE_FILES = _case_files()
CASE_IDS = [str(p.relative_to(CASES_DIR)) for p in CASE_FILES]


def test_corpus_has_exactly_fifty_cases():
    assert len(CASE_FILES) == 50


def test_corpus_matches_the_category_distribution_exactly():
    counts = {}
    for path in CASE_FILES:
        category = path.parent.name
        counts[category] = counts.get(category, 0) + 1
    assert counts == EXPECTED_DISTRIBUTION


def test_every_case_id_is_globally_unique():
    ids = [_load(p)["id"] for p in CASE_FILES]
    assert len(set(ids)) == len(ids), "duplicate case id(s) found"


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_case_has_every_required_top_level_key(path):
    data = _load(path)
    missing = REQUIRED_TOP_LEVEL_KEYS - set(data)
    assert not missing, f"{path}: missing {missing}"


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_case_id_matches_its_filename(path):
    data = _load(path)
    assert data["id"] == path.stem


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_case_category_matches_its_directory(path):
    data = _load(path)
    assert data["category"] == path.parent.name


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_case_id_is_prefixed_for_its_category(path):
    """The <category>_<nn>_<slug> naming convention -- every id starts with the right short
    prefix for its directory, so a case can never silently drift into the wrong category."""
    prefixes = {
        "happy_path": "hp_", "ambiguity": "amb_", "identity_scoping": "id_",
        "authorization": "auth_", "policy": "pol_", "dirty_data": "dd_",
        "hallucination": "hal_", "over_escalation": "over_", "provisional": "prov_",
    }
    data = _load(path)
    assert data["id"].startswith(prefixes[data["category"]])


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_case_has_at_least_one_scripted_turn(path):
    data = _load(path)
    assert isinstance(data["turns"], list) and len(data["turns"]) >= 1
    assert all(isinstance(t, str) and t.strip() for t in data["turns"])


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_case_persona_and_principal_type_agree(path):
    data = _load(path)
    assert data["persona"] in ("C", "S")
    principal = data["principal"]
    assert principal["type"] in ("customer", "staff")
    if principal["type"] == "staff":
        assert principal.get("role") in ("dispatcher", "manager", "owner")
        assert data["persona"] == "S"
    else:
        assert data["persona"] == "C"


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_if_attempted_guard_has_tool_and_decision(path):
    """`if_attempted` may be a single {tool, decision[, reason]} dict or a list of them --
    either way, every entry needs at least tool and decision."""
    data = _load(path)
    if_attempted = data.get("guards", {}).get("if_attempted")
    if if_attempted is None:
        return
    specs = if_attempted if isinstance(if_attempted, list) else [if_attempted]
    for spec in specs:
        assert {"tool", "decision"} <= set(spec)


def test_every_authorization_case_scores_attack_outcome():
    for path in CASE_FILES:
        if path.parent.name != "authorization":
            continue
        data = _load(path)
        assert data.get("scored", {}).get("attack_outcome") is True, path


def test_only_auth_10_is_flagged_for_substitution_replay():
    flagged = [p.stem for p in CASE_FILES if _load(p).get("substitution_replay")]
    assert flagged == ["auth_10_oracle_probe_sequence_C"]


def test_every_customer_principal_id_resolves_in_the_golden_db(edge_db):
    with get_session() as session:
        real_ids = {c.id for c in session.query(Customer.id).all()}
    for path in CASE_FILES:
        data = _load(path)
        principal = data["principal"]
        if principal["type"] == "customer" and principal["id"] is not None:
            assert principal["id"] in real_ids, f"{path}: customer id {principal['id']} not seeded"
