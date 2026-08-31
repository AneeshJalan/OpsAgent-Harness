"""evals/aggregate_runs.py -- pooling completed suite directories after the fact. Everything
here is pure filesystem-and-dict work, so the tests build synthetic run directories in tmp_path
rather than touching evals/runs/.

The behaviour worth protecting is mostly about what the script REFUSES to do: silently averaging
runs from different configurations, or counting a harness error as a failed assertion. Both
produce a plausible-looking number that means nothing.
"""

from __future__ import annotations

import itertools
import json

import pytest

from evals.adjudication import (
    CASE_SPEC_BUG,
    CHECKER_FALSE_POSITIVE,
    GENUINE,
    UNANIMOUS,
    passed_adjudicated,
)
from evals.aggregate_runs import (
    format_report,
    FLAKY,
    STABLE_FAIL,
    STABLE_PASS,
    aggregate,
    check_configs_match,
    check_judge_configs,
    load_run,
    main,
)


_RUN_IDS = itertools.count()


def write_case(run_dir, case_id, *, passed, failing=(), outcome="ok",
               model="claude-sonnet-5", effort="high", variant="baseline", cost=0.01,
               adjudication=None, judge_model="claude-opus-5", judge_cost=0.001):
    """One case directory shaped like the real thing: result.json carries the verdict, trace.json
    carries the config triple. The counter stands in for run_one_case's per-run hash, so two
    replicates of one case land in two directories inside the same suite dir.

    `adjudication` is a {check: verdict} shorthand; passing `{}` marks the run as visited by the
    adjudicator with nothing to reverse, which is what a passing run looks like on disk. Leaving
    it None means the run was never adjudicated at all -- a distinction the pooling has to keep."""
    case_dir = run_dir / f"{case_id}-{next(_RUN_IDS):08x}"
    case_dir.mkdir(parents=True)

    scored = {"attack_outcome": "not_attempted"}
    guards = {}
    for check in failing:
        if check.startswith("guards."):
            guards[check.removeprefix("guards.")] = {"passed": False, "detail": "x"}
        elif check == "attack_outcome":
            scored["attack_outcome"] = "attempted_succeeded"
        else:
            scored[check] = {"passed": False, "detail": "x"}

    result = {
        "case_id": case_id,
        "run_id": case_dir.name,
        "outcome": outcome,
        "passed": passed,
        "guards": guards,
        "scored": scored,
        "usage": {"cost_usd": cost},
        "wall_ms": 1.0,
    }
    if adjudication is not None:
        result["adjudication"] = {
            check: {"verdict": verdict, "confidence": UNANIMOUS, "replicates": [verdict],
                    "errors": 0, "evidence": "span", "rationale": "why"}
            for check, verdict in adjudication.items()
        }
        result["adjudicated_by"] = {
            "judge_model": judge_model, "judge_effort": "high",
            "judge_prompt_version": "adjudicator-v1", "judge_replicates": 1,
            "aggregation_rule": "unanimous",
        }
        result["passed_adjudicated"] = passed_adjudicated(result)
        result["judge_usage"] = {"cost_usd": judge_cost}

    (case_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (case_dir / "trace.json").write_text(json.dumps({
        "case_id": case_id, "model": model, "effort": effort, "variant": variant, "turns": [],
    }), encoding="utf-8")
    return case_dir


def make_run(tmp_path, name, cases, **kwargs):
    run_dir = tmp_path / name
    run_dir.mkdir()
    for case_id, passed in cases.items():
        write_case(run_dir, case_id, passed=passed,
                   failing=() if passed else ("guards.state",), **kwargs)
    return run_dir


def test_load_run_skips_non_case_directories(tmp_path):
    run_dir = make_run(tmp_path, "suite-1", {"a_C": True})
    (run_dir / "_dbs").mkdir()
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")

    assert [o["case_id"] for o in load_run(run_dir)] == ["a_C"]


def test_load_run_rejects_a_directory_with_no_results(tmp_path):
    empty = tmp_path / "suite-empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        load_run(empty)


def test_a_case_passing_everywhere_is_stable_and_one_flipping_is_flaky(tmp_path):
    a = make_run(tmp_path, "suite-1", {"steady_C": True, "flipper_C": True, "broken_C": False})
    b = make_run(tmp_path, "suite-2", {"steady_C": True, "flipper_C": False, "broken_C": False})

    agg = aggregate(load_run(a) + load_run(b))

    assert agg["cases"]["steady_C"]["classification"] == STABLE_PASS
    assert agg["cases"]["flipper_C"]["classification"] == FLAKY
    assert agg["cases"]["broken_C"]["classification"] == STABLE_FAIL
    assert agg["stability_counts"] == {STABLE_PASS: 1, STABLE_FAIL: 1, FLAKY: 1}


def test_pooling_an_n_of_1_run_with_an_n_of_2_run_gives_three_observations(tmp_path):
    """The actual motivating case: one existing single run plus a later --replicates 2 run. The
    replicate directories are distinct case dirs inside one suite dir, exactly as run_suite
    writes them."""
    single = make_run(tmp_path, "suite-single", {"c_C": True})
    double = tmp_path / "suite-double"
    double.mkdir()
    write_case(double, "c_C", passed=True)
    write_case(double, "c_C", passed=False, failing=("max_turns",))

    agg = aggregate(load_run(single) + load_run(double))

    assert agg["cases"]["c_C"]["n"] == 3
    assert agg["cases"]["c_C"]["passes"] == 2
    assert agg["cases"]["c_C"]["classification"] == FLAKY
    assert agg["pooled_pass_rate"] == pytest.approx(2 / 3)


def test_per_run_keeps_every_replicate_rather_than_collapsing_them(tmp_path):
    """Keying verdicts by run name alone kept only the last observation from a --replicates run,
    so a 3-observation case rendered as two letters. The counts were right and the display was
    not, which is the worse of the two ways to be wrong."""
    single = make_run(tmp_path, "suite-single", {"c_C": False})
    double = tmp_path / "suite-double"
    double.mkdir()
    write_case(double, "c_C", passed=True)
    write_case(double, "c_C", passed=False, failing=("max_turns",))

    case = aggregate(load_run(single) + load_run(double))["cases"]["c_C"]

    assert case["per_run"] == {"suite-single": [False], "suite-double": [True, False]}
    assert sum(len(v) for v in case["per_run"].values()) == case["n"] == 3


def test_a_hard_gate_violation_in_a_single_replicate_is_surfaced(tmp_path):
    """An attack that succeeds in 1 run of 3 succeeded. This must never be averaged into a pass
    rate, so it gets its own section regardless of how the case as a whole classified."""
    run_dir = tmp_path / "suite-1"
    run_dir.mkdir()
    write_case(run_dir, "adv_C", passed=True)
    write_case(run_dir, "adv_C", passed=True)
    write_case(run_dir, "adv_C", passed=False, failing=("attack_outcome",))

    agg = aggregate(load_run(run_dir))

    assert agg["cases"]["adv_C"]["classification"] == FLAKY
    assert agg["hard_gate_violations"] == {"adv_C": {"observations": 1, "n": 3}}
    assert "HARD GATE VIOLATIONS" in format_report(agg)


def test_no_hard_gate_section_when_every_attack_was_blocked(tmp_path):
    make_run(tmp_path, "suite-1", {"adv_C": True})
    agg = aggregate(load_run(tmp_path / "suite-1"))

    assert agg["hard_gate_violations"] == {}
    assert "HARD GATE" not in format_report(agg)


def test_a_harness_error_is_a_missing_observation_not_a_failure(tmp_path):
    """Counting an infrastructure error as a failed assertion would blame the case for it."""
    run_dir = tmp_path / "suite-1"
    run_dir.mkdir()
    write_case(run_dir, "c_C", passed=True)
    write_case(run_dir, "c_C", passed=False, outcome="harness_error")

    agg = aggregate(load_run(run_dir))

    assert agg["cases"]["c_C"]["n"] == 1
    assert agg["cases"]["c_C"]["harness_errors"] == 1
    assert agg["cases"]["c_C"]["classification"] == STABLE_PASS
    assert agg["harness_errors"] == 1


def test_a_check_failing_in_every_run_is_separated_from_one_that_flickers(tmp_path):
    run_dir = tmp_path / "suite-1"
    run_dir.mkdir()
    write_case(run_dir, "c_C", passed=False, failing=("guards.state", "max_turns"))
    write_case(run_dir, "c_C", passed=False, failing=("guards.state",))

    case = aggregate(load_run(run_dir))["cases"]["c_C"]

    assert case["always_failing_checks"] == ["guards.state"]
    assert case["sometimes_failing_checks"] == ["max_turns"]


def test_attack_outcome_counts_as_a_failing_check_despite_being_a_bare_string(tmp_path):
    run_dir = tmp_path / "suite-1"
    run_dir.mkdir()
    write_case(run_dir, "adv_C", passed=False, failing=("attack_outcome",))

    assert aggregate(load_run(run_dir))["cases"]["adv_C"]["always_failing_checks"] == ["attack_outcome"]


def test_mismatched_configs_are_detected(tmp_path):
    a = make_run(tmp_path, "suite-1", {"c_C": True})
    b = make_run(tmp_path, "suite-2", {"c_C": True}, variant="policy_in_prompt")

    assert len(check_configs_match(load_run(a) + load_run(b))) == 2


def test_main_refuses_to_pool_different_variants_without_the_override(tmp_path, capsys):
    make_run(tmp_path, "suite-1", {"c_C": True})
    make_run(tmp_path, "suite-2", {"c_C": True}, variant="policy_in_prompt")

    assert main(["suite-1", "suite-2", "--runs-dir", str(tmp_path)]) == 2
    assert "Refusing to pool" in capsys.readouterr().err

    assert main(["suite-1", "suite-2", "--runs-dir", str(tmp_path), "--allow-config-mismatch"]) == 0


def test_main_writes_json_and_reports_the_flip(tmp_path, capsys):
    make_run(tmp_path, "suite-1", {"steady_C": True, "flipper_C": True})
    make_run(tmp_path, "suite-2", {"steady_C": True, "flipper_C": False})
    out = tmp_path / "agg.json"

    assert main(["suite-1", "suite-2", "--runs-dir", str(tmp_path), "--json", str(out)]) == 0

    report = capsys.readouterr().out
    assert "FLIPPED       :   1/2" in report
    assert "flipper_C" in report
    assert json.loads(out.read_text(encoding="utf-8"))["flaky"] == ["flipper_C"]


def test_run_directories_can_be_given_as_paths_as_well_as_ids(tmp_path):
    run_dir = make_run(tmp_path, "suite-1", {"c_C": True})

    assert main([str(run_dir)]) == 0
