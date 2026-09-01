"""CLI: merges two or more completed suite directories under evals/runs/ into one reliability
report, after the fact and at zero API cost.

The motivating problem: a single pass over the suite gives one observation per case, and this
agent is non-deterministic enough that ~24% of cases changed verdict between two consecutive
runs of nearly the same build. A pass rate computed from n=1 cannot distinguish "the harness
change worked" from "the model sampled differently", which makes it useless for the thing it is
usually quoted for. `run_suite --replicates N` fixes that going forward, but re-running a suite
you have already paid for is pure waste -- an existing single run is already a valid replicate
of any later run of the same build, so this script pools them instead.

Usage:
    python -m evals.aggregate_runs suite-1788082276 suite-1788090000
    python -m evals.aggregate_runs suite-a suite-b --json aggregate.json

What it will not do: pool runs whose model, effort or variant differ. Those are different
experiments and averaging them produces a number that describes no build at all, so it is a
hard error rather than a warning (--allow-config-mismatch overrides it for the deliberate case
of comparing a variant against itself).

Adjudication adds a second configuration axis, and it deliberately does NOT get the same
treatment. Which model adjudicated a run, under which prompt and which aggregation rule, decides
how its *failures* were re-scored -- but not what conversation happened. Refusing to merge two
runs with identical agent config merely because one was adjudicated and the other wasn't would
destroy the entire point of this script, which is pooling runs already paid for. So a judge-config
mismatch degrades instead of refusing: the deterministic rate is still pooled over everything, and
the adjudicated block is either suppressed with the reason printed, or pooled explicitly with
--allow-judge-mismatch. Partial adjudication is reported with both denominators and never
imputed -- counting an un-adjudicated run as "adjudicated, no reversals" would bias the
adjudicated rate downward over runs the adjudicator never saw.

The build itself is the one compatibility condition this script CANNOT check: the trace records
model/effort/variant but not the git SHA of the harness that produced it. Pooling runs from
different builds silently mixes an old checker's verdicts with a new one's. Until the SHA is
recorded, that check is the caller's responsibility, and the report prints the run directories
it pooled so the claim can at least be audited afterwards.

Output is deliberately a *stability classification*, not a tighter pass rate. At n=2 or n=3 per
case there is no meaningful confidence interval to be had, but there is a very useful three-way
split: cases that always pass, cases that always fail, and cases that flip. Only the third group
explains a moving suite-level number, and only the first two are safe to draw conclusions from.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.adjudication import (
    CHECKER_FALSE_POSITIVE,
    GENUINE,
    INSUFFICIENT_EVIDENCE,
    adjudicated_failing_checks,
    failing_checks,
    is_reversal,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"

STABLE_PASS = "stable_pass"
STABLE_FAIL = "stable_fail"
FLAKY = "flaky"
NO_OBSERVATIONS = "no_valid_observations"

# The judge fields that decide whether two adjudicated runs are comparable. Not the same axis as
# the agent's model/effort/variant: these change how failures were re-scored, not what
# conversation happened.
_JUDGE_CONFIG_KEYS = ("judge_model", "judge_effort", "judge_prompt_version",
                      "judge_replicates", "aggregation_rule")


def _judge_config_label(result: dict[str, Any]) -> str | None:
    """One printable string per distinct judge configuration, or None for a run nothing
    adjudicated. Built from the fields the adjudicator stamps onto each result it writes."""
    config = result.get("adjudicated_by")
    if not isinstance(config, dict):
        return None
    return " ".join(f"{key}={config.get(key)}" for key in _JUDGE_CONFIG_KEYS)


def load_run(run_dir: Path, *, use_adjudication: bool = True) -> list[dict[str, Any]]:
    """One observation per case directory. Reads trace.json purely for the config triple --
    result.json does not carry it, and pooling runs blind to it is the main way this script
    could produce a wrong answer.

    Adjudication data is read from the same result.json when present. `adjudicated` records
    whether the run was *visited*, which is not the same as whether anything was reversed: a
    passing run is visited, has an empty verdict map, and still belongs in the adjudicated
    denominator. `use_adjudication=False` drops all of it, which is what --adjudication none
    does."""
    if not run_dir.is_dir():
        raise SystemExit(f"not a directory: {run_dir}")

    observations = []
    for case_dir in sorted(run_dir.iterdir()):
        result_path = case_dir / "result.json"
        if not result_path.is_file():
            continue  # _dbs/ and any other non-case directory
        result = json.loads(result_path.read_text(encoding="utf-8"))

        config = {"model": None, "effort": None, "variant": None}
        trace_path = case_dir / "trace.json"
        if trace_path.is_file():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            if isinstance(trace, dict):
                config = {key: trace.get(key) for key in config}

        adjudication = result.get("adjudication") if use_adjudication else None
        observations.append({
            "run": run_dir.name,
            "case_id": result["case_id"],
            "run_id": result.get("run_id"),
            "outcome": result.get("outcome"),
            "passed": result.get("passed"),
            "failing_checks": failing_checks(result),
            "cost_usd": (result.get("usage") or {}).get("cost_usd", 0.0),
            "adjudicated": adjudication is not None,
            "passed_adjudicated": result.get("passed_adjudicated") if use_adjudication else None,
            "adjudicated_failing_checks": (
                adjudicated_failing_checks(result, adjudication) if adjudication is not None
                else None
            ),
            "adjudication": adjudication or {},
            "judge_config": _judge_config_label(result) if use_adjudication else None,
            "judge_cost_usd": (
                (result.get("judge_usage") or {}).get("cost_usd", 0.0) if use_adjudication else 0.0
            ),
            **config,
        })

    if not observations:
        raise SystemExit(f"no case results found under {run_dir}")
    return observations


def check_configs_match(observations: list[dict[str, Any]]) -> list[str]:
    """Returns a human-readable list of the distinct config triples present. One entry means the
    pool is coherent; more than one means the caller is averaging different experiments."""
    seen = {(o["model"], o["effort"], o["variant"]) for o in observations}
    return sorted(f"model={m} effort={e} variant={v}" for m, e, v in seen)


def check_judge_configs(observations: list[dict[str, Any]]) -> list[str]:
    """The distinct judge configurations present among the *adjudicated* observations. Empty when
    nothing was adjudicated; more than one entry means two different judges' verdicts are about to
    be averaged into a single adjudicated rate, which is the same error as merging two agent
    models, one level up."""
    return sorted({o["judge_config"] for o in observations
                   if o["adjudicated"] and o["judge_config"]})


def _classify(passes: int, n: int) -> str:
    if n == 0:
        return NO_OBSERVATIONS
    if passes == n:
        return STABLE_PASS
    if passes == 0:
        return STABLE_FAIL
    return FLAKY


def _adjudicated_block(
    observations: list[dict[str, Any]], cases: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """The parallel set of numbers, or None when nothing in the pool was adjudicated.

    The delta is computed against the deterministic rate **over the adjudicated subset**, not
    against the headline deterministic rate. When coverage is complete those are the same number;
    when it isn't, differencing rates over different observation sets would produce an artifact
    rather than a finding. Both denominators are reported either way, so the gap is never
    something a reader has to infer.

    Judge replicates never appear here. They were already reduced to one verdict per
    (case, check, observation) before anything was written to disk, so the sampling unit is the
    agent observation and nothing about re-running the judge can inflate it."""
    scored = [o for o in observations if o["outcome"] == "ok"]
    adjudicated = [o for o in scored if o["adjudicated"]]
    if not adjudicated:
        return None

    det_same_subset = sum(1 for o in adjudicated if o["passed"]) / len(adjudicated)
    adj_rate = sum(1 for o in adjudicated if o["passed_adjudicated"]) / len(adjudicated)

    reversals = 0
    # `unresolved` is not a verdict -- it is the judge failing to return one at all (an API error,
    # or a reply with no verdict in it). Kept apart from `insufficient_evidence`, which IS a
    # verdict: one says the judge could not answer, the other says the judge answered that the
    # evidence could not settle it. Collapsing them would hide an infrastructure problem inside a
    # measurement.
    verdict_counts = {GENUINE: 0, CHECKER_FALSE_POSITIVE: 0, INSUFFICIENT_EVIDENCE: 0,
                      "unresolved": 0}
    by_check: dict[str, dict[str, int]] = defaultdict(
        lambda: {"failures": 0, GENUINE: 0, CHECKER_FALSE_POSITIVE: 0, INSUFFICIENT_EVIDENCE: 0,
                 "reversed": 0, "unresolved": 0})
    instability = {"unanimous": 0, "majority": 0, "split": 0, "unresolved": 0}

    for obs in adjudicated:
        reversals += len(set(obs["failing_checks"]) - set(obs["adjudicated_failing_checks"] or []))
        for check, entry in sorted((obs["adjudication"] or {}).items()):
            stats = by_check[check]
            stats["failures"] += 1
            verdict = entry.get("verdict")
            key = verdict if verdict in verdict_counts else "unresolved"
            verdict_counts[key] += 1
            stats[key] += 1
            instability[entry.get("confidence") or "unresolved"] += 1
            if is_reversal(entry):
                stats["reversed"] += 1

    # The finding, not the delta: cases that fail every time deterministically and pass every time
    # once checker false positives are reversed. Restricted to fully adjudicated cases, because a
    # case whose second replicate was never looked at has not been shown to recover.
    recovered = sorted(
        case_id for case_id, case in cases.items()
        if case["classification"] == STABLE_FAIL
        and case["adjudicated_classification"] == STABLE_PASS
        and case["adjudicated_n"] == case["n"]
    )
    partial = sorted(
        case_id for case_id, case in cases.items()
        if 0 < case["adjudicated_n"] < case["n"]
    )

    return {
        "adjudicated_observations": len(adjudicated),
        "scored_observations": len(scored),
        "complete_coverage": len(adjudicated) == len(scored),
        "deterministic_pass_rate_over_adjudicated": det_same_subset,
        "pooled_pass_rate": adj_rate,
        "delta_pts": (adj_rate - det_same_subset) * 100,
        "reversals": reversals,
        "verdict_counts": verdict_counts,
        "by_check": {k: dict(v) for k, v in sorted(by_check.items())},
        "judge_instability": instability,
        "recovered_by_adjudication": recovered,
        "partially_adjudicated_cases": partial,
        "judge_configs": check_judge_configs(observations),
        "judge_cost_usd": sum(o["judge_cost_usd"] or 0.0 for o in adjudicated),
    }


def aggregate(observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Pools observations by case_id.

    Two pass rates are reported and they answer different questions. `pooled_pass_rate` is the
    fraction of all *runs* that passed -- comparable to a single run's number, and the one to
    quote. `mean_case_pass_rate` averages each case's own rate first, so every case counts once
    regardless of how many times it was sampled; the two diverge only when replicate counts are
    uneven, and a gap between them means the pool is unbalanced.

    No confidence interval is computed. Observations of the same case are not independent of
    each other, so a binomial interval over the flat observation list would be anti-conservative
    -- and at n<=3 per case, the honest summary is the stability split, not an interval."""
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        by_case[obs["case_id"]].append(obs)

    cases = {}
    for case_id, obs_list in sorted(by_case.items()):
        # A harness error is not a failed assertion -- it is a missing observation, and counting
        # it as a failure would silently punish a case for an infrastructure problem.
        scored = [o for o in obs_list if o["outcome"] == "ok"]
        passes = sum(1 for o in scored if o["passed"])
        n = len(scored)
        classification = _classify(passes, n)

        # The same three-way split computed a second time, over the adjudicated subset only. A
        # case adjudicated in one replicate but not the other is classified on what was actually
        # adjudicated, and `adjudicated_n` is carried alongside so the gap stays visible rather
        # than being silently filled in.
        adjudicated = [o for o in scored if o["adjudicated"]]
        adj_n = len(adjudicated)
        adj_passes = sum(1 for o in adjudicated if o["passed_adjudicated"])
        adj_classification = _classify(adj_passes, adj_n) if adj_n else None

        # Checks that fired in some runs but not all are what make a case flaky; a check that
        # fails every time is a stable finding even when the case as a whole flips.
        check_counts: dict[str, int] = defaultdict(int)
        for o in scored:
            for check in o["failing_checks"]:
                check_counts[check] += 1

        reversed_here = sorted({
            check for o in adjudicated
            for check in set(o["failing_checks"]) - set(o["adjudicated_failing_checks"] or [])
        })

        cases[case_id] = {
            "n": n,
            "passes": passes,
            "pass_rate": passes / n if n else None,
            "classification": classification,
            "adjudicated_n": adj_n,
            "adjudicated_passes": adj_passes,
            "adjudicated_classification": adj_classification,
            "reversed_checks": reversed_here,
            "harness_errors": len(obs_list) - n,
            "always_failing_checks": sorted(c for c, k in check_counts.items() if k == n),
            "sometimes_failing_checks": sorted(c for c, k in check_counts.items() if 0 < k < n),
            "check_failure_counts": dict(sorted(check_counts.items())),
            # A list per run, not a single verdict: a --replicates N run contributes N
            # observations from one suite directory, and keying them by run name alone silently
            # kept only the last one.
            "per_run": {
                run: [o["passed"] for o in obs_list if o["run"] == run]
                for run in sorted({o["run"] for o in obs_list})
            },
        }

    scored_all = [o for o in observations if o["outcome"] == "ok"]
    per_run: dict[str, dict[str, Any]] = {}
    for run in sorted({o["run"] for o in observations}):
        run_obs = [o for o in scored_all if o["run"] == run]
        per_run[run] = {
            "cases": len(run_obs),
            "passed": sum(1 for o in run_obs if o["passed"]),
            "pass_rate": (sum(1 for o in run_obs if o["passed"]) / len(run_obs)) if run_obs else None,
        }

    rates = [c["pass_rate"] for c in cases.values() if c["pass_rate"] is not None]
    classifications = [c["classification"] for c in cases.values()]

    return {
        "runs_pooled": sorted({o["run"] for o in observations}),
        "configs": check_configs_match(observations),
        "total_observations": len(observations),
        "scored_observations": len(scored_all),
        "harness_errors": len(observations) - len(scored_all),
        "distinct_cases": len(cases),
        "pooled_pass_rate": (
            sum(1 for o in scored_all if o["passed"]) / len(scored_all) if scored_all else None
        ),
        "mean_case_pass_rate": (sum(rates) / len(rates)) if rates else None,
        "per_run": per_run,
        "stable_pass": sorted(c for c, v in cases.items() if v["classification"] == STABLE_PASS),
        "stable_fail": sorted(c for c, v in cases.items() if v["classification"] == STABLE_FAIL),
        "flaky": sorted(c for c, v in cases.items() if v["classification"] == FLAKY),
        # Surfaced separately because it is the one number that must be zero, and because it is
        # the one check a pass rate must never be allowed to average away: an attack that
        # succeeds in 1 run of 3 succeeded. Sampling more only ever finds more of these.
        "hard_gate_violations": {
            case_id: {"observations": case["check_failure_counts"]["attack_outcome"], "n": case["n"]}
            for case_id, case in cases.items()
            if case["check_failure_counts"].get("attack_outcome")
        },
        "stability_counts": {
            STABLE_PASS: classifications.count(STABLE_PASS),
            STABLE_FAIL: classifications.count(STABLE_FAIL),
            FLAKY: classifications.count(FLAKY),
        },
        "total_cost_usd": sum(o["cost_usd"] or 0.0 for o in observations),
        # Present only when something in the pool was adjudicated. Everything above is untouched
        # by it: the deterministic rate stays the headline number and means exactly what it always
        # meant.
        "adjudicated": _adjudicated_block(observations, cases),
        "cases": cases,
    }


def format_report(agg: dict[str, Any]) -> str:
    lines = []
    add = lines.append

    add(f"Pooled {len(agg['runs_pooled'])} run(s): {', '.join(agg['runs_pooled'])}")
    add(f"Config: {'; '.join(agg['configs'])}")
    add(f"{agg['distinct_cases']} distinct cases, {agg['scored_observations']} scored observations"
        + (f" ({agg['harness_errors']} harness errors excluded)" if agg["harness_errors"] else ""))
    add("")

    add("Pass rate per run:")
    for run, stats in agg["per_run"].items():
        add(f"  {run:24s} {stats['passed']:3d}/{stats['cases']:<3d} = {stats['pass_rate']:.3f}")
    add(f"  {'POOLED':24s} {'':7s}   {agg['pooled_pass_rate']:.3f}")
    add("")

    adj = agg.get("adjudicated")
    if adj:
        det = adj["deterministic_pass_rate_over_adjudicated"]
        add("Adjudicated (checker false positives reversed, unanimous verdicts only):")
        for config in adj["judge_configs"]:
            add(f"  {config}")
        add(f"  deterministic {det:.3f} -> adjudicated {adj['pooled_pass_rate']:.3f}"
            f"   {adj['delta_pts']:+.1f} pts over {adj['adjudicated_observations']} observations")
        if not adj["complete_coverage"]:
            # Printed, never imputed: treating the un-adjudicated remainder as "no reversals"
            # would bias the adjudicated rate downward over runs the adjudicator never saw.
            add(f"  NOTE: only {adj['adjudicated_observations']} of "
                f"{adj['scored_observations']} scored observations were adjudicated; the delta is"
                f" computed over that subset alone.")
        if adj["partially_adjudicated_cases"]:
            add(f"  {len(adj['partially_adjudicated_cases'])} case(s) adjudicated in some"
                f" replicates but not all: {', '.join(adj['partially_adjudicated_cases'])}")
        counts = adj["verdict_counts"]
        add(f"  verdicts: {counts[CHECKER_FALSE_POSITIVE]} false positive,"
            f" {counts[GENUINE]} genuine, {counts[INSUFFICIENT_EVIDENCE]} undetermined,"
            f" {counts['unresolved']} unresolved"
            f"   ({adj['reversals']} checks actually reversed)")
        inst = adj["judge_instability"]
        add(f"  judge self-consistency: {inst['unanimous']} unanimous,"
            f" {inst['majority']} majority, {inst['split']} split")
        if adj["recovered_by_adjudication"]:
            add("  RECOVERED (always failed deterministically, always passes adjudicated):")
            for case_id in adj["recovered_by_adjudication"]:
                checks = ", ".join(agg["cases"][case_id]["reversed_checks"])
                add(f"    {case_id:46s} {checks}")
        else:
            add("  recovered by adjudication: none")
        add(f"  judge cost: ${adj['judge_cost_usd']:.4f}")
        add("")

    hard = agg["hard_gate_violations"]
    if hard:
        add("HARD GATE VIOLATIONS (must be zero -- an attack that succeeded in any run succeeded):")
        for case_id, stats in sorted(hard.items()):
            add(f"  {case_id:48s} {stats['observations']}/{stats['n']} observations")
        add("")

    counts = agg["stability_counts"]
    total = agg["distinct_cases"]
    add("Stability (the number that actually matters at this sample size):")
    add(f"  always passed : {counts[STABLE_PASS]:3d}/{total}")
    add(f"  always failed : {counts[STABLE_FAIL]:3d}/{total}")
    add(f"  FLIPPED       : {counts[FLAKY]:3d}/{total}")
    if total:
        add(f"  -> {counts[FLAKY] / total:.1%} of cases are unreliable at this sample size; a"
            f" suite-level move smaller than that is not evidence of anything.")
    add("")

    if agg["flaky"]:
        add("Flaky cases (per-run verdict, then which checks were unstable):")
        for case_id in agg["flaky"]:
            case = agg["cases"][case_id]
            # Grouped by run, so "F PF" reads as one observation in the first run and two in the
            # second -- the replicate structure stays visible instead of being flattened.
            verdicts = " ".join(
                "".join("P" if p else "F" for p in run_verdicts)
                for run_verdicts in case["per_run"].values()
            )
            add(f"  {case_id:48s} {case['passes']}/{case['n']}  [{verdicts}]")
            if case["sometimes_failing_checks"]:
                add(f"      unstable: {', '.join(case['sometimes_failing_checks'])}")
            if case["always_failing_checks"]:
                add(f"      always:   {', '.join(case['always_failing_checks'])}")
        add("")

    if agg["stable_fail"]:
        add("Always-failing cases (safe to act on -- these are real, repeatable findings):")
        for case_id in agg["stable_fail"]:
            case = agg["cases"][case_id]
            add(f"  {case_id:48s} {', '.join(case['always_failing_checks']) or '(mixed checks)'}")
        add("")

    add(f"Pooled API cost of the runs merged here: ${agg['total_cost_usd']:.2f} (this merge cost $0)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge completed suite runs into one reliability report, at no API cost.")
    parser.add_argument("runs", nargs="+",
                        help="suite run ids under evals/runs/, or paths to run directories")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--json", type=Path, default=None,
                        help="also write the full aggregate as JSON to this path")
    parser.add_argument("--allow-config-mismatch", action="store_true",
                        help="pool runs whose model/effort/variant differ (they are different "
                             "experiments; the pooled rate will describe neither)")
    parser.add_argument("--adjudication", choices=("auto", "none"), default="auto",
                        help="auto: use adjudication verdicts wherever they are on disk "
                             "(default). none: ignore them and report the deterministic rate "
                             "alone")
    parser.add_argument("--allow-judge-mismatch", action="store_true",
                        help="pool observations adjudicated under different judge configurations "
                             "into one adjudicated rate; without it the adjudicated block is "
                             "suppressed and the reason printed")
    args = parser.parse_args(argv)

    observations: list[dict[str, Any]] = []
    for name in args.runs:
        candidate = Path(name)
        run_dir = candidate if candidate.is_dir() else args.runs_dir / name
        observations.extend(load_run(run_dir, use_adjudication=args.adjudication == "auto"))

    configs = check_configs_match(observations)
    if len(configs) > 1 and not args.allow_config_mismatch:
        print("Refusing to pool runs with different configurations:", file=sys.stderr)
        for config in configs:
            print(f"  {config}", file=sys.stderr)
        print("Pass --allow-config-mismatch if this is deliberate.", file=sys.stderr)
        return 2

    # Judge mismatch degrades rather than refusing. The agent config decides what conversation
    # happened and a mismatch there invalidates the whole pool; the judge config only decides how
    # failures were re-scored, so the deterministic rate below is still perfectly good and
    # refusing to print it would throw away the runs this script exists to merge.
    judge_configs = check_judge_configs(observations)
    if len(judge_configs) > 1 and not args.allow_judge_mismatch:
        print("Two or more judge configurations in this pool; suppressing the adjudicated rate:",
              file=sys.stderr)
        for config in judge_configs:
            print(f"  {config}", file=sys.stderr)
        print("Pass --allow-judge-mismatch to pool them anyway. The deterministic rate below is "
              "unaffected.", file=sys.stderr)
        for obs in observations:
            obs["adjudicated"] = False

    agg = aggregate(observations)
    print(format_report(agg))

    if args.json:
        args.json.write_text(json.dumps(agg, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
