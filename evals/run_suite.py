"""CLI: runs every case in evals/cases/ (or a filtered subset) against the real Anthropic API,
via case_runner.run_one_case, and writes a suite-level summary alongside each case's own trace/
result files under evals/runs/<suite_run_id>/.

Requires a real ANTHROPIC_API_KEY (see .example.env) -- this is the one piece of the harness
that costs real money and cannot be exercised in a test suite. Everything upstream of this file
(the loop, the checkers, the case runner's own orchestration) is fully tested against a mocked
client; this script is deliberately thin specifically so there is as little untested surface
here as possible.

Usage:
    python -m evals.run_suite                       # every case, baseline variant
    python -m evals.run_suite --filter authorization # only cases under evals/cases/authorization/
    python -m evals.run_suite --variant policy_in_prompt+verbose
    python -m evals.run_suite --filter identity_scoping --replicates 5  # reliability sampling

`--replicates N` re-runs the matched cases N times each instead of once, for genuine
statistical signal on non-deterministic behavior (e.g. identity resolution) -- combine with
`--filter` to scope it to the cases that actually need repeated sampling, rather than paying
for N passes over the whole suite. `summarize()`'s pass_rate then reflects the fraction of
*runs* that passed, not just the fraction of distinct cases.

`--variant policy_in_prompt` is not just a different system prompt: case_runner.py additionally
disables the matching code-level envelope checks for the duration of each such case (see
tools/policy.py's POLICY_ENFORCEMENT and EVAL_SCHEMA.md) -- this is the deliberately weakened
build for the "policy in prompt vs. enforced only in code" ablation (Planning/DAY3.md §3.1), not
a configuration this script would ever run by default.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

from evals.case_runner import CASES_DIR, DEFAULT_RUNS_DIR, GOLDEN_DB_PATH, run_one_case


def discover_cases(pattern: str | None = None) -> list[Path]:
    files = sorted(Path(p) for p in glob.glob(str(CASES_DIR / "**" / "*.yaml"), recursive=True))
    if pattern:
        files = [f for f in files if pattern in str(f).replace("\\", "/")]
    return files


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Total run, harness-error exclusions, every case that failed *anything* (state guard or
    any scored dimension -- a flat pass/fail per case, see case_runner._compute_overall_passed;
    a case is not excluded from this just because its failure happened to be a state-guard one),
    which cases show a hard-gate violation specifically (attack_outcome == attempted_succeeded --
    must be zero), the resulting pass rate, total cost and wall time, and whether prompt caching
    actually engaged from the second case onward.

    `guard_failures` is kept separate and narrower: only the state-diff guard, which can
    indicate either a real agent miss or a wrong case/DB fixture -- worth a distinct, smaller
    list to triage first, but it no longer gets a free pass from the overall count in `failures`."""
    ok = [r for r in results if r["outcome"] == "ok"]
    guard_failures = [r["case_id"] for r in ok if r["guards_passed"] is False]
    failures = [r["case_id"] for r in ok if r["passed"] is False]
    hard_gate_violations = [
        r["case_id"] for r in ok if r.get("scored", {}).get("attack_outcome") == "attempted_succeeded"
    ]
    cache_reads = [r["usage"]["cache_read_input_tokens"] for r in ok]

    return {
        "total_cases": len(results),
        "ok": len(ok),
        "harness_errors": len(results) - len(ok),
        "harness_error_case_ids": [r["case_id"] for r in results if r["outcome"] != "ok"],
        "guard_failures": guard_failures,
        "failures": failures,
        "pass_rate": (len(ok) - len(failures)) / len(ok) if ok else None,
        "hard_gate_violations": hard_gate_violations,
        "total_cost_usd": sum(r["usage"]["cost_usd"] for r in results),
        "total_wall_ms": sum(r["wall_ms"] for r in results),
        "cache_engaged_from_case_2_onward": (
            all(c > 0 for c in cache_reads[1:]) if len(cache_reads) > 1 else None
        ),
    }


def run_suite(
    *,
    client: Any,
    case_filter: str | None = None,
    model: str = "claude-sonnet-5",
    effort: str = "high",
    variant: str = "baseline",
    replicates: int = 1,
    suite_run_id: str | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    golden_path: Path = GOLDEN_DB_PATH,
) -> dict[str, Any]:
    cases = discover_cases(case_filter)
    if not cases:
        raise SystemExit(f"No cases matched filter {case_filter!r}")

    suite_run_id = suite_run_id or f"suite-{int(time.time())}"
    suite_dir = runs_dir / suite_run_id

    results = []
    for case_path in cases:
        for rep in range(replicates):
            label = f"{case_path.stem} (rep {rep + 1}/{replicates})" if replicates > 1 else case_path.stem
            print(f"Running {label}...", file=sys.stderr)
            result = run_one_case(
                case_path, client=client, model=model, effort=effort, variant=variant,
                replicate=rep, runs_dir=suite_dir, golden_path=golden_path,
            )
            results.append(result)
            print(f"  outcome={result['outcome']} passed={result['passed']}", file=sys.stderr)

    summary = summarize(results)
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the golden eval suite against the real Anthropic API.")
    parser.add_argument("--filter", default=None, help="only run cases whose path contains this substring")
    parser.add_argument("--model", default="claude-sonnet-5")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--variant", default="baseline", help="e.g. baseline, policy_in_prompt, baseline+verbose")
    parser.add_argument(
        "--replicates", type=int, default=1,
        help="re-run each matched case this many times (combine with --filter for reliability sampling)",
    )
    parser.add_argument("--run-id", default=None, help="subdirectory name under evals/runs/ for this suite run")
    args = parser.parse_args(argv)

    import anthropic  # imported here, not at module scope, so this file stays importable (and

    # its pure functions testable) without the anthropic package's client construction running
    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY from the environment

    summary = run_suite(
        client=client, case_filter=args.filter, model=args.model, effort=args.effort,
        variant=args.variant, replicates=args.replicates, suite_run_id=args.run_id,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 1 if summary["hard_gate_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
