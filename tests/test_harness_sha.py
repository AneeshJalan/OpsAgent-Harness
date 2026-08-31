"""Recording which build of the harness produced a run, and warning when a pool mixes two.

The trace has always recorded what the *model* was. It never recorded what the *checkers* were,
and a pass rate is as much a property of the checkers as of the agent -- so pooling a run scored
before a checker fix with one scored after it produced a number describing neither, with nothing
in either directory to say so.

The warning is a warning on purpose. Comparing across checker builds is a legitimate thing to
want; doing it without noticing is not.
"""

from __future__ import annotations

import json

from agent.trace import Trace, current_harness_sha
from evals.aggregate_runs import aggregate, check_harness_shas, format_report, load_run
from tests.test_aggregate_runs import make_run, write_case


def test_the_sha_is_a_string_or_none_and_never_raises():
    # None when git is unavailable or this is not a checkout. A missing provenance stamp must
    # never be able to fail a run that was otherwise fine.
    sha = current_harness_sha()
    assert sha is None or (isinstance(sha, str) and sha)


def test_a_trace_carries_the_build_it_was_produced_by():
    trace = Trace(run_id="r", case_id="c", persona="C", variant="baseline",
                  model="claude-sonnet-5", effort="high", replicate=0, principal={})
    assert "harness_sha" in trace.to_dict()
    assert trace.to_dict()["harness_sha"] == current_harness_sha()


def test_an_explicit_sha_survives_a_round_trip(tmp_path):
    trace = Trace(run_id="r", case_id="c", persona="C", variant="baseline",
                  model="claude-sonnet-5", effort="high", replicate=0, principal={},
                  harness_sha="abc1234")
    path = trace.write(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["harness_sha"] == "abc1234"


def write_run(tmp_path, name, sha):
    run_dir = tmp_path / name
    run_dir.mkdir()
    write_case(run_dir, "a_C", passed=True)
    trace_path = next(run_dir.glob("a_C-*")) / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if sha is not None:
        trace["harness_sha"] = sha
    trace_path.write_text(json.dumps(trace), encoding="utf-8")
    return run_dir


def test_one_build_is_named_in_the_header(tmp_path):
    agg = aggregate(load_run(write_run(tmp_path, "suite-1", "abc1234")))
    assert agg["harness_shas"] == ["abc1234"]
    assert "Harness build: abc1234" in format_report(agg)


def test_two_builds_warn_but_still_report(tmp_path):
    # A warning, not a refusal: the caller may well be comparing checker builds deliberately.
    obs = load_run(write_run(tmp_path, "suite-1", "abc1234"))
    obs += load_run(write_run(tmp_path, "suite-2", "def5678"))
    report = format_report(aggregate(obs))

    assert "pooled across 2 harness builds" in report
    assert "abc1234, def5678" in report
    assert "POOLED" in report  # the rest of the report still prints


def test_a_run_written_before_the_sha_existed_reads_as_unknown(tmp_path):
    obs = load_run(write_run(tmp_path, "suite-1", None))
    assert check_harness_shas(obs) == ["unknown"]
    # Not the same as matching, and shown as such once a second build joins the pool.
    obs += load_run(write_run(tmp_path, "suite-2", "abc1234"))
    assert check_harness_shas(obs) == ["abc1234", "unknown"]
    assert "pooled across 2 harness builds" in format_report(aggregate(obs))


def test_the_agent_config_check_is_unaffected_by_the_sha(tmp_path):
    # The SHA is a fourth field on the config dict but must not leak into the model/effort/variant
    # triple, whose mismatch is still a hard error.
    obs = load_run(write_run(tmp_path, "suite-1", "abc1234"))
    obs += load_run(write_run(tmp_path, "suite-2", "def5678"))
    assert aggregate(obs)["configs"] == ["model=claude-sonnet-5 effort=high variant=baseline"]


def test_pooling_still_works_when_no_trace_records_a_sha(tmp_path):
    agg = aggregate(load_run(make_run(tmp_path, "suite-1", {"a_C": True, "b_C": False})))
    assert agg["harness_shas"] == ["unknown"]
    assert "Harness build: unknown" in format_report(agg)
