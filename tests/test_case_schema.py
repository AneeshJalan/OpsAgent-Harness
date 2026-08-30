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
REQUIRED_TOP_LEVEL_KEYS = {
    "id", "category", "persona", "risks", "escalation_expected", "db", "principal", "turns", "guards", "scored",
}

# The planned category distribution -- exact category -> count this corpus must match.
EXPECTED_DISTRIBUTION = {
    "happy_path": 8,
    "ambiguity": 6,
    "identity_scoping": 7,
    "authorization": 10,
    "policy": 6,
    "dirty_data": 6,
    "hallucination": 4,
    "over_escalation": 3,
    "provisional": 2,
    "adversarial": 18,
}


def _case_files() -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(CASES_DIR / "**" / "*.yaml"), recursive=True))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


CASE_FILES = _case_files()
CASE_IDS = [str(p.relative_to(CASES_DIR)) for p in CASE_FILES]


def test_corpus_has_the_expected_case_count():
    assert len(CASE_FILES) == sum(EXPECTED_DISTRIBUTION.values())


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
        "adversarial": "adv_",
    }
    data = _load(path)
    assert data["id"].startswith(prefixes[data["category"]])


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_escalation_expected_is_a_valid_ground_truth_label(path):
    """R8/R9's confusion matrix needs a ground truth per case: was a
    callback/auto-escalation genuinely the correct outcome here, was it genuinely not, or is
    escalation simply not what this case is testing at all. `na` is not a shrug -- most of this
    corpus (identity, dirty data, authorization) is legitimately silent on the question, and
    conflating "not applicable" with "false" would corrupt the precision/recall denominator."""
    data = _load(path)
    assert data["escalation_expected"] in (True, False, "na")


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


# --- guards.state vs. what the tools actually write ---------------------------------------
#
# Tables each tool writes on the path a case expects, excluding audit_log (which check_state
# always ignores). Derived by executing every tool against a fresh copy of the golden DB and
# diffing the snapshots, not by reading the code -- that is how the indirect writes got caught:
# create_invoice, apply_discount and record_payment all call recompute_balances(), so they
# rewrite `customers` as well as the table they obviously touch, and request_human_callback
# queues a pending_requests row.
#
# If a tool's side effects change, these tests fail loudly. That is deliberate: the cases'
# state guards have to be revisited when they do.

_READ_ONLY = {
    "find_my_account", "list_services", "get_availability", "get_my_appointments", "get_quote",
    "get_payment_link", "search_customers", "get_customer_detail", "list_appointments",
    "get_schedule", "list_invoices", "list_technicians", "find_duplicate_candidates",
    "find_schedule_conflicts",
}


def _tables_written(tool: str, principal_resolved: bool, decision: str | None) -> set[str]:
    """The tables `tool` certainly writes, given how the case expects it to resolve."""
    d = decision or "executed"
    if tool in _READ_ONLY:
        return set()
    if tool == "book_appointment":
        written = set() if principal_resolved else {"customers"}
        return written | ({"pending_requests"} if d == "queued" else {"appointments"})
    if tool in ("reschedule_appointment", "book_appointment_for_customer"):
        return {"pending_requests"} if d == "queued" else {"appointments"}
    if tool == "apply_discount":
        return {"pending_requests"} if d == "queued" else {"invoices", "invoice_lines", "customers"}
    if tool == "create_invoice":
        return {"invoices", "invoice_lines", "customers"}
    if tool == "record_payment":
        return {"invoices", "customers"}
    if tool == "send_invoice":
        return {"invoices"}
    if tool == "add_internal_note":
        return {"customers"}
    if tool in ("cancel_appointment", "cancel_appointment_with_notice", "reassign_technician"):
        return {"appointments"}
    if tool in ("write_off_balance", "void_invoice", "merge_customers", "request_human_callback"):
        return {"pending_requests"}
    raise AssertionError(f"no side-effect model for tool {tool!r} -- add one")


def _expected_mutators(data: dict) -> dict[str, str | None]:
    """Tools the case asserts will actually run, mapped to the decision it expects from them."""
    selection = (data.get("scored", {}) or {}).get("selection", {}) or {}
    out: dict[str, str | None] = {t: None for t in selection.get("require_tools", [])}
    rd = selection.get("require_decision")
    if rd and rd["decision"] != "denied":
        out[rd["tool"]] = rd["decision"]
    if_attempted = (data.get("guards", {}) or {}).get("if_attempted") or []
    if isinstance(if_attempted, dict):
        if_attempted = [if_attempted]
    for spec in if_attempted:
        if spec["decision"] in ("executed", "queued"):
            out[spec["tool"]] = spec["decision"]
    return {t: d for t, d in out.items() if t not in _READ_ONLY}


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_state_guard_is_not_contradicted_by_the_tools_the_case_requires(path):
    """A case cannot both require a write to happen and assert nothing changed. id_07 shipped
    with `state: {unchanged: true}` alongside require_decision{reschedule_appointment: executed},
    which no agent could satisfy."""
    data = _load(path)
    state = (data.get("guards", {}) or {}).get("state") or {}
    mutators = _expected_mutators(data)
    if state.get("unchanged"):
        assert not mutators, (
            f"{path}: state.unchanged=true but the case requires {sorted(mutators)} to run"
        )


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_state_guard_names_every_table_its_required_tools_write(path):
    """check_state fails a case outright on any change to a table its `tables` block does not
    name. hp_07 declared invoices and invoice_lines but not customers, so the correct
    create-then-send path always tripped 'unexpected changes'."""
    data = _load(path)
    state = (data.get("guards", {}) or {}).get("state") or {}
    declared = set((state.get("tables") or {}).keys())
    if not declared:
        return
    resolved = data["principal"].get("id") is not None or data["persona"] == "S"
    for tool, decision in _expected_mutators(data).items():
        uncovered = _tables_written(tool, resolved, decision) - declared
        assert not uncovered, (
            f"{path}: {tool} also writes {sorted(uncovered)}, which guards.state.tables does "
            f"not name -- check_state will report 'unexpected changes'"
        )


# --- on_confirmation_request ---------------------------------------------------------------


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_confirmation_affordance_is_a_non_empty_string_when_present(path):
    data = _load(path)
    if "on_confirmation_request" not in data:
        return
    reply = data["on_confirmation_request"]
    assert isinstance(reply, str) and reply.strip(), (
        f"{path}: on_confirmation_request must be a non-empty string"
    )


@pytest.mark.parametrize("path", CASE_FILES, ids=CASE_IDS)
def test_confirmation_affordance_is_not_used_to_smuggle_in_a_scripted_turn(path):
    """It answers a question the agent asked; it is not a place to put the next thing the caller
    wanted to say. A case needing another instruction should add a `turns` entry, which plays
    unconditionally and is therefore honest about being part of the script."""
    data = _load(path)
    reply = data.get("on_confirmation_request")
    if reply is None:
        return
    assert len(reply) <= 200, f"{path}: on_confirmation_request reads like a scripted turn"


def test_only_cases_that_actually_stall_carry_an_affordance():
    """Kept as an explicit allow-list rather than a free-for-all: every entry here was verified
    against a real trace that ended on an unanswered confirmation. Adding one to a case that
    does not stall makes the suite easier to pass without making the agent any better, so a new
    entry should come with the trace that justifies it."""
    expected = {
        "pol_01_after_hours_C", "pol_02_lead_time_C", "pol_03_booking_window_C",
        "pol_06_no_skilled_tech_C", "hal_01_queued_not_done_C",
        "prov_01_fall_forward_small_booking_C",
    }
    actual = {
        _load(p)["id"] for p in CASE_FILES if _load(p).get("on_confirmation_request")
    }
    assert actual == expected
