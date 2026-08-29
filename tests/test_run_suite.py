"""evals/run_suite.py -- discover_cases and summarize are pure functions, fully testable without
an API key. run_suite() itself is exercised end to end against a handful of real case files, but
with an AutoEndTurnClient (tests/fakes.py) standing in for the real API -- this proves the
looping/aggregation wiring, not any individual case's correctness (that's test_case_runner.py's
job with a precisely scripted client).
"""

from __future__ import annotations

import json

from evals.run_suite import discover_cases, run_suite, summarize
from fakes import AutoEndTurnClient


def test_discover_cases_finds_every_case_with_no_filter():
    # Derived from the same distribution test_case_schema.py enforces, rather than a second
    # hardcoded total that has to be remembered every time a category is added.
    from test_case_schema import EXPECTED_DISTRIBUTION

    assert len(discover_cases()) == sum(EXPECTED_DISTRIBUTION.values())


def test_discover_cases_filter_narrows_to_one_category():
    matches = discover_cases("dirty_data")
    assert len(matches) == 6
    assert all("dirty_data" in str(p).replace("\\", "/") for p in matches)


def test_discover_cases_filter_can_match_a_single_case_id():
    matches = discover_cases("hp_01_book_standard_C")
    assert len(matches) == 1


def _result(
    case_id, outcome="ok", guards_passed=True, passed=True, attack_outcome=None,
    cache_read=100, cost=0.01, wall_ms=500,
):
    scored = {}
    if attack_outcome is not None:
        scored["attack_outcome"] = attack_outcome
    return {
        "case_id": case_id, "outcome": outcome, "guards_passed": guards_passed, "passed": passed,
        "scored": scored,
        "usage": {"cache_read_input_tokens": cache_read, "cost_usd": cost},
        "wall_ms": wall_ms,
    }


def test_summarize_counts_ok_and_harness_errors():
    results = [_result("a"), _result("b", outcome="harness_error", guards_passed=None, passed=None)]
    summary = summarize(results)
    assert summary["total_cases"] == 2
    assert summary["ok"] == 1
    assert summary["harness_errors"] == 1
    assert summary["harness_error_case_ids"] == ["b"]


def test_summarize_lists_guard_failures():
    results = [_result("a", guards_passed=True), _result("b", guards_passed=False)]
    summary = summarize(results)
    assert summary["guard_failures"] == ["b"]


def test_summarize_lists_failures_for_any_failed_check_not_just_state_guards():
    """The gap this rework closes: a case whose only problem is a failed scored dimension (no
    state guard involved at all) must still show up as a failure at the suite level -- not just
    be silently absent because guard_failures only ever tracked the state guard."""
    results = [
        _result("clean", guards_passed=True, passed=True),
        _result("state_guard_only", guards_passed=False, passed=False),
        _result("scored_only", guards_passed=True, passed=False),  # the case this test exists for
    ]
    summary = summarize(results)
    assert summary["guard_failures"] == ["state_guard_only"]  # unchanged, narrow signal
    assert set(summary["failures"]) == {"state_guard_only", "scored_only"}  # the comprehensive one


def test_summarize_pass_rate():
    results = [_result("a", passed=True), _result("b", passed=True), _result("c", passed=False)]
    summary = summarize(results)
    assert summary["pass_rate"] == 2 / 3


def test_summarize_pass_rate_is_none_with_no_ok_cases():
    results = [_result("a", outcome="harness_error", guards_passed=None, passed=None)]
    assert summarize(results)["pass_rate"] is None


def test_summarize_flags_hard_gate_violations():
    results = [
        _result("auth_ok", attack_outcome="attempted_blocked"),
        _result("auth_bad", attack_outcome="attempted_succeeded"),
    ]
    summary = summarize(results)
    assert summary["hard_gate_violations"] == ["auth_bad"]


def test_summarize_totals_cost_and_wall_time():
    results = [_result("a", cost=0.01, wall_ms=500), _result("b", cost=0.02, wall_ms=700)]
    summary = summarize(results)
    assert summary["total_cost_usd"] == 0.03
    assert summary["total_wall_ms"] == 1200


def test_summarize_cache_check_true_when_all_but_first_are_nonzero():
    results = [_result("a", cache_read=0), _result("b", cache_read=500), _result("c", cache_read=800)]
    assert summarize(results)["cache_engaged_from_case_2_onward"] is True


def test_summarize_cache_check_false_when_a_later_case_never_hit_cache():
    results = [_result("a", cache_read=0), _result("b", cache_read=0), _result("c", cache_read=800)]
    assert summarize(results)["cache_engaged_from_case_2_onward"] is False


def test_summarize_cache_check_is_none_with_fewer_than_two_ok_cases():
    assert summarize([_result("a")])["cache_engaged_from_case_2_onward"] is None


def test_run_suite_wires_discovery_through_to_a_written_summary(tmp_path, edge_db_with_policy):
    client = AutoEndTurnClient()
    summary = run_suite(
        client=client, case_filter="hp_02_quote_published_price_C",
        golden_path=edge_db_with_policy, runs_dir=tmp_path, suite_run_id="test-suite",
    )
    assert summary["total_cases"] == 1
    assert summary["ok"] == 1

    summary_path = tmp_path / "test-suite" / "summary.json"
    assert summary_path.exists()
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["total_cases"] == 1

    # the individual case's own result.json was written too, inside the suite's run directory
    result_files = list((tmp_path / "test-suite").glob("hp_02_quote_published_price_C-*/result.json"))
    assert len(result_files) == 1


def test_run_suite_replicates_reruns_each_matched_case_n_times(tmp_path, edge_db_with_policy):
    """--replicates is the reliability-sampling mechanism from C11: combined with --filter, it
    re-runs a case multiple times instead of once, so pass_rate reflects genuine repeated
    sampling rather than a single pass/fail per case."""
    client = AutoEndTurnClient()
    summary = run_suite(
        client=client, case_filter="hp_02_quote_published_price_C", replicates=3,
        golden_path=edge_db_with_policy, runs_dir=tmp_path, suite_run_id="test-suite-reps",
    )
    assert summary["total_cases"] == 3
    assert summary["ok"] == 3

    result_files = list((tmp_path / "test-suite-reps").glob("hp_02_quote_published_price_C-*/result.json"))
    assert len(result_files) == 3  # each replicate got its own run_id/result.json, none clobbered


def test_run_suite_raises_clearly_on_an_empty_filter(tmp_path, edge_db_with_policy):
    import pytest

    with pytest.raises(SystemExit):
        run_suite(
            client=AutoEndTurnClient(), case_filter="no_such_category_xyz",
            golden_path=edge_db_with_policy, runs_dir=tmp_path,
        )
