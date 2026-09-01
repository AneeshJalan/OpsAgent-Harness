"""evals/aggregate_runs.py -- the adjudicated half of pooling. Shares test_aggregate_runs.py's
synthetic run directories, which now grow optional adjudication verdicts.

Two rules these tests exist to protect. The deterministic rate is never touched by any of this: it
is the number every existing table was computed from, and adjudication is strictly additive
alongside it. And the adjudicated rate is never computed over a denominator it does not own --
partial coverage is reported with both denominators rather than imputed, because counting an
un-adjudicated run as "adjudicated, no reversals" would make the adjudicator look less useful than
it is on runs it never saw.
"""

from __future__ import annotations

import pytest

from evals.adjudication import CHECKER_FALSE_POSITIVE, GENUINE, INSUFFICIENT_EVIDENCE
from evals.aggregate_runs import (
    aggregate,
    check_configs_match,
    check_judge_configs,
    format_report,
    load_run,
    main,
)
from tests.test_aggregate_runs import make_run, write_case


def adjudicated_run(tmp_path, name, **kwargs):
    """One run holding all four shapes that matter: a passing case, a case recovered by
    adjudication, a case the adjudicator agreed was a genuine failure, and a case failing an exact
    check nothing can reverse."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    write_case(run_dir, "hap_C", passed=True, adjudication={}, **kwargs)
    write_case(run_dir, "soft_fp_C", passed=False, failing=["must_not_contain"],
               adjudication={"must_not_contain": CHECKER_FALSE_POSITIVE}, **kwargs)
    write_case(run_dir, "soft_real_C", passed=False, failing=["no_pii"],
               adjudication={"no_pii": GENUINE}, **kwargs)
    write_case(run_dir, "exact_C", passed=False, failing=["require_tools"],
               adjudication={}, **kwargs)
    return run_dir


def test_the_deterministic_rate_is_unchanged_by_adjudication(tmp_path):
    agg = aggregate(load_run(adjudicated_run(tmp_path, "suite-1")))
    assert agg["pooled_pass_rate"] == pytest.approx(1 / 4)
    assert agg["stable_fail"] == ["exact_C", "soft_fp_C", "soft_real_C"]


def test_the_adjudicated_rate_is_reported_beside_it(tmp_path):
    agg = aggregate(load_run(adjudicated_run(tmp_path, "suite-1")))
    adj = agg["adjudicated"]

    assert adj["adjudicated_observations"] == 4
    assert adj["complete_coverage"] is True
    assert adj["deterministic_pass_rate_over_adjudicated"] == pytest.approx(1 / 4)
    assert adj["pooled_pass_rate"] == pytest.approx(2 / 4)
    assert adj["delta_pts"] == pytest.approx(25.0)
    assert adj["reversals"] == 1
    assert adj["verdict_counts"][CHECKER_FALSE_POSITIVE] == 1
    assert adj["verdict_counts"][GENUINE] == 1


def test_recovered_cases_are_named_with_the_checks_that_moved(tmp_path):
    # This list is the finding, not the delta: the per-case answer to where the judge beat the
    # deterministic checkers.
    agg = aggregate(load_run(adjudicated_run(tmp_path, "suite-1")))

    assert agg["adjudicated"]["recovered_by_adjudication"] == ["soft_fp_C"]
    assert agg["cases"]["soft_fp_C"]["reversed_checks"] == ["must_not_contain"]
    assert "RECOVERED" in format_report(agg)


def test_an_undetermined_verdict_does_not_recover_a_case(tmp_path):
    # Not knowing whether a failure is real is not knowing it is not. It is counted on its own
    # axis so a question nobody could answer is never read as a checker that was right.
    run_dir = tmp_path / "suite-1"
    run_dir.mkdir()
    write_case(run_dir, "unclear_C", passed=False, failing=["grounding"],
               adjudication={"grounding": INSUFFICIENT_EVIDENCE})
    agg = aggregate(load_run(run_dir))

    assert agg["adjudicated"]["pooled_pass_rate"] == pytest.approx(0.0)
    assert agg["adjudicated"]["recovered_by_adjudication"] == []
    assert agg["adjudicated"]["verdict_counts"][INSUFFICIENT_EVIDENCE] == 1
    assert agg["adjudicated"]["verdict_counts"][GENUINE] == 0
    assert "undetermined" in format_report(agg)


def test_an_unadjudicated_pool_has_no_adjudicated_block(tmp_path):
    agg = aggregate(load_run(make_run(tmp_path, "suite-1", {"a_C": True, "b_C": False})))
    assert agg["adjudicated"] is None
    assert "Adjudicated" not in format_report(agg)


def test_partial_coverage_is_reported_with_both_denominators(tmp_path):
    run_dir = tmp_path / "suite-1"
    run_dir.mkdir()
    write_case(run_dir, "soft_fp_C", passed=False, failing=["must_not_contain"],
               adjudication={"must_not_contain": CHECKER_FALSE_POSITIVE})
    write_case(run_dir, "never_seen_C", passed=False, failing=["must_not_contain"])
    agg = aggregate(load_run(run_dir))
    adj = agg["adjudicated"]

    assert adj["adjudicated_observations"] == 1
    assert adj["scored_observations"] == 2
    assert adj["complete_coverage"] is False
    # Computed over the adjudicated subset alone. Differencing rates over different observation
    # sets produces an artifact rather than a finding.
    assert adj["deterministic_pass_rate_over_adjudicated"] == pytest.approx(0.0)
    assert adj["pooled_pass_rate"] == pytest.approx(1.0)
    assert "only 1 of 2 scored observations were adjudicated" in format_report(agg)


def test_a_case_adjudicated_in_only_one_replicate_does_not_count_as_recovered(tmp_path):
    run_dir = tmp_path / "suite-1"
    run_dir.mkdir()
    write_case(run_dir, "half_C", passed=False, failing=["must_not_contain"],
               adjudication={"must_not_contain": CHECKER_FALSE_POSITIVE})
    write_case(run_dir, "half_C", passed=False, failing=["must_not_contain"])
    agg = aggregate(load_run(run_dir))

    assert agg["cases"]["half_C"]["adjudicated_n"] == 1
    assert agg["cases"]["half_C"]["n"] == 2
    assert agg["adjudicated"]["recovered_by_adjudication"] == []
    assert agg["adjudicated"]["partially_adjudicated_cases"] == ["half_C"]


def test_judge_configs_are_listed_separately_from_agent_configs(tmp_path):
    obs = load_run(adjudicated_run(tmp_path, "suite-1"))
    assert check_configs_match(obs) == ["model=claude-sonnet-5 effort=high variant=baseline"]
    assert check_judge_configs(obs) == [
        "judge_model=claude-opus-5 judge_effort=high judge_prompt_version=adjudicator-v1 "
        "judge_replicates=1 aggregation_rule=unanimous"
    ]


def test_two_judge_configs_suppress_the_adjudicated_rate_but_not_the_run(tmp_path, capsys):
    # Degrade, do not refuse. Refusing to merge two runs with identical agent config merely
    # because they were adjudicated differently would destroy the point of this script.
    a = adjudicated_run(tmp_path, "suite-1")
    b = adjudicated_run(tmp_path, "suite-2", judge_model="claude-sonnet-5")

    assert main([str(a), str(b)]) == 0
    captured = capsys.readouterr()
    assert "suppressing the adjudicated rate" in captured.err
    assert "Pooled 2 run(s)" in captured.out
    assert "Adjudicated (" not in captured.out


def test_allow_judge_mismatch_pools_them_and_names_both(tmp_path, capsys):
    a = adjudicated_run(tmp_path, "suite-1")
    b = adjudicated_run(tmp_path, "suite-2", judge_model="claude-sonnet-5")

    assert main([str(a), str(b), "--allow-judge-mismatch"]) == 0
    out = capsys.readouterr().out
    assert "Adjudicated (" in out
    assert "judge_model=claude-opus-5" in out and "judge_model=claude-sonnet-5" in out


def test_adjudication_none_ignores_verdicts_on_disk(tmp_path, capsys):
    assert main([str(adjudicated_run(tmp_path, "suite-1")), "--adjudication", "none"]) == 0
    out = capsys.readouterr().out
    assert "Adjudicated (" not in out
    assert "0.250" in out


def test_judge_cost_is_kept_out_of_the_agent_cost_total(tmp_path):
    # The existing cost tables quote total_cost_usd. Folding judge spend into it would make every
    # historical cost-per-successful-task figure quietly wrong.
    agg = aggregate(load_run(adjudicated_run(tmp_path, "suite-1")))
    assert agg["total_cost_usd"] == pytest.approx(0.04)
    assert agg["adjudicated"]["judge_cost_usd"] == pytest.approx(0.004)
