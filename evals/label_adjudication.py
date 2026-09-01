"""CLI: hand-label the adjudicable failures in a completed suite, blind, so the adjudicator's
agreement with a person can be measured rather than assumed.

These labels are a HELD-OUT REFERENCE SET. They are never shown to the adjudicator -- not as
few-shot examples, not as anything -- and nothing in evals/adjudicate.py reads the file this
writes. That separation is the entire point: labels the judge has seen cannot measure the judge,
any more than test data a model trained on can. If few-shot examples are ever wanted in the
adjudicator's prompt, they have to be drawn from a disjoint set of labels, not from this one.

Order of operations:

    1. python -m evals.label_adjudication suite-1788088729          # label, blind, before judging
    2. python -m evals.adjudicate         suite-1788088729 ...      # the judge never sees step 1
    3. python -m evals.label_adjudication suite-1788088729 --score  # compare the two

Labelling before judging is not a convention, it is what makes the comparison worth anything: a
label written after reading the judge's answer is anchored to it, and agreement measured that way
is agreement with yourself.

One item per distinct (case, check) pair rather than per failing observation. Two replicates of the
same case failing the same check are not independent items, and counting them twice would make the
sample look larger and every interval tighter than it is. The pair is labelled on one nominated
run, and that run's id is recorded, so the comparison in step 3 is against the verdict the judge
reached on the same transcript.

The transcript shown is the one the JUDGE gets -- same elision budget, same hidden reasoning. Show
a person more than the model saw and any disagreement conflates "the judge reasons worse" with "the
judge was given less", which are different findings with different fixes. `--full` lifts the budget
for a deliberate second look, and records that it was lifted.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from evals.adjudicate import CHECK_CLAIMS, _DEFAULT_MAX_RESULT_CHARS, case_dirs, render_transcript
from evals.adjudication import (
    CHECKER_FALSE_POSITIVE,
    GENUINE,
    INSUFFICIENT_EVIDENCE,
    VERDICTS,
    adjudicable_failures,
)

DEFAULT_RUNS_DIR = Path(__file__).resolve().parent / "runs"
LABELS_FILENAME = "hand_labels.json"

# Single keystroke per verdict, in the order a labeller will most often want them. `s` defers an
# item without recording anything, which is not the same as `insufficient_evidence`: one says
# "come back to this", the other is a considered judgment that the evidence cannot settle it.
_KEYS = {
    "1": GENUINE,
    "2": CHECKER_FALSE_POSITIVE,
    "3": INSUFFICIENT_EVIDENCE,
}
_MENU = (
    "  [1] genuine                 the checker was right; the transcript really does show this\n"
    "  [2] checker_false_positive  the checker was wrong; the transcript does not show this\n"
    "  [3] insufficient_evidence   cannot tell from what is here\n"
    "  [r] re-show transcript   [n] add a note   [s] skip for now   [q] save and quit"
)


def labelling_queue(suite_dir: Path) -> list[dict[str, Any]]:
    """One entry per distinct (case_id, check) adjudicable failure in the suite.

    Where several observations of a case fail the same check, the first is nominated and the rest
    are counted in `observations`. Nominating rather than merging keeps the item tied to a real
    transcript -- there is no such thing as the average of two conversations, and a label has to be
    about something a person actually read."""
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for case_dir in case_dirs(suite_dir):
        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        if result.get("outcome") != "ok":
            continue
        scored = result.get("scored") or {}
        for check in adjudicable_failures(result):
            key = (result["case_id"], check)
            if key in seen:
                seen[key]["observations"] += 1
                continue
            seen[key] = {
                "case_id": result["case_id"],
                "run_id": result.get("run_id"),
                "check": check,
                "detail": (scored.get(check) or {}).get("detail", ""),
                "case_dir": str(case_dir),
                "observations": 1,
            }
    return [seen[key] for key in sorted(seen)]


def label_key(item: dict[str, Any]) -> str:
    return f"{item['run_id']}::{item['check']}"


def load_labels(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"suite": None, "labels": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("labels", {})
    return data


def save_labels(path: Path, data: dict[str, Any]) -> None:
    """Written after every single label, not at the end. A labelling pass is an hour of human
    attention and the session will be interrupted; losing it to a closed terminal is avoidable."""
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def render_item(item: dict[str, Any], index: int, total: int, budget: int) -> str:
    """The question, exactly as the judge will be asked it, plus the transcript it will read.

    Deliberately does NOT show the case's other failures, whether the case passed overall, or any
    verdict the adjudicator may already have recorded. The first two are context the judge is also
    denied; the third would destroy the blindness this whole file exists to preserve."""
    trace_path = Path(item["case_dir"]) / "trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    replicates = (f"   ({item['observations']} observations of this pair; labelling "
                  f"{item['run_id']})" if item["observations"] > 1 else "")

    return "\n".join([
        "=" * 100,
        f"[{index}/{total}]  {item['case_id']}   check: {item['check']}{replicates}",
        "=" * 100,
        "",
        f"WHAT THE CHECK CLAIMS: {CHECK_CLAIMS.get(item['check'], '(no description)')}",
        f"CHECKER DETAIL:        {item['detail'] or '(none recorded)'}",
        "",
        "-" * 100,
        render_transcript(trace, budget),
        "-" * 100,
        "",
        "Does this transcript exhibit the defect the check claims?",
        _MENU,
    ])


def run_session(
    suite_dir: Path,
    *,
    labels_path: Path,
    budget: int = _DEFAULT_MAX_RESULT_CHARS,
    seed: int = 0,
    limit: int | None = None,
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> dict[str, Any]:
    """The interactive loop. Returns the labels file's contents.

    Items are shuffled under a recorded seed. Left in discovery order they arrive grouped by case
    id, which means grouped by category -- all the adversarial cases, then all the ambiguity ones --
    and a labeller's standards drift over an hour. Shuffling spreads that drift evenly across
    checks instead of confounding it with them. The seed is recorded so the order is reproducible.

    Already-labelled items are skipped, so the session is resumable."""
    queue = labelling_queue(suite_dir)
    random.Random(seed).shuffle(queue)
    if limit is not None:
        queue = queue[:limit]

    data = load_labels(labels_path)
    data["suite"] = suite_dir.name
    data["shuffle_seed"] = seed
    data["max_result_chars"] = budget
    data.setdefault("started_at", datetime.now(timezone.utc).isoformat())

    pending = [item for item in queue if label_key(item) not in data["labels"]]
    if not pending:
        write(f"All {len(queue)} item(s) already labelled in {labels_path}.")
        return data

    write(f"{len(queue)} distinct (case, check) pairs; {len(pending)} left to label.")
    write("Labels are held out -- the adjudicator never sees this file.\n")

    for offset, item in enumerate(pending, start=1):
        prompt = render_item(item, offset, len(pending), budget)
        write(prompt)
        note = ""
        while True:
            answer = (read("verdict> ") or "").strip().lower()
            if answer == "q":
                save_labels(labels_path, data)
                write(f"\nSaved {len(data['labels'])} label(s) to {labels_path}.")
                return data
            if answer == "s":
                break
            if answer == "r":
                write(prompt)
                continue
            if answer == "n":
                note = (read("note> ") or "").strip()
                continue
            if answer in _KEYS:
                data["labels"][label_key(item)] = {
                    "case_id": item["case_id"],
                    "run_id": item["run_id"],
                    "check": item["check"],
                    "verdict": _KEYS[answer],
                    "note": note,
                    "labelled_at": datetime.now(timezone.utc).isoformat(),
                }
                save_labels(labels_path, data)
                write(f"  recorded: {_KEYS[answer]}\n")
                break
            write("  unrecognised -- press 1, 2, 3, r, n, s or q")

    save_labels(labels_path, data)
    write(f"\nDone. {len(data['labels'])} label(s) in {labels_path}.")
    return data


def cohens_kappa(pairs: list[tuple[str, str]]) -> dict[str, Any]:
    """Cohen's kappa over (human, judge) label pairs, with the observed agreement it corrects.

    Returns kappa=None when chance agreement is total (p_e == 1), which happens whenever both
    raters used exactly one category -- the formula is 0/0 there, and reporting 0.0 would say
    "no better than chance" about data showing perfect agreement. That is not an edge case to be
    tidied away: at small n with a skewed class it is a live outcome, and it is the clearest signal
    that the sample cannot support the statistic."""
    n = len(pairs)
    if n == 0:
        return {"n": 0, "observed_agreement": None, "expected_agreement": None, "kappa": None}

    agree = sum(1 for a, b in pairs if a == b)
    p_o = agree / n
    human = Counter(a for a, _ in pairs)
    judge = Counter(b for _, b in pairs)
    p_e = sum((human[c] / n) * (judge[c] / n) for c in set(human) | set(judge))

    out: dict[str, Any] = {
        "n": n,
        "agreements": agree,
        "observed_agreement": p_o,
        "expected_agreement": p_e,
        "kappa": None,
        "se": None,
        "ci95": None,
    }
    if p_e >= 1.0:
        out["undefined_because"] = "chance agreement is total; both raters used a single category"
        return out

    kappa = (p_o - p_e) / (1 - p_e)
    se = math.sqrt(p_o * (1 - p_o) / (n * (1 - p_e) ** 2)) if p_o not in (0.0, 1.0) else 0.0
    out["kappa"] = kappa
    out["se"] = se
    out["ci95"] = [max(-1.0, kappa - 1.96 * se), min(1.0, kappa + 1.96 * se)]
    return out


def score(suite_dir: Path, labels_path: Path) -> dict[str, Any]:
    """Compare the held-out labels against whatever the adjudicator recorded.

    The headline is a BINARY collapse -- "is this failure genuine?" -- because that is the question
    the pass rate actually turns on, and because a 3-way kappa at this sample size spreads the data
    over nine cells that mostly hold zero. The 3x3 matrix is reported underneath for the
    undetermined split.

    Per-check agreement is reported as raw counts with no per-check kappa. At one to five items per
    check a kappa is not a weak estimate, it is frequently undefined, and publishing eight of them
    would invite exactly the over-reading the binary headline avoids."""
    data = load_labels(labels_path)
    labels = data.get("labels") or {}
    if not labels:
        raise SystemExit(f"no labels in {labels_path} -- run without --score first")

    judged: dict[str, str] = {}
    for case_dir in case_dirs(suite_dir):
        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        for check, entry in (result.get("adjudication") or {}).items():
            if isinstance(entry, dict) and entry.get("verdict"):
                judged[f"{result.get('run_id')}::{check}"] = entry["verdict"]

    pairs, missing = [], []
    per_check: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "agree": 0})
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(VERDICTS, 0))

    for key, label in sorted(labels.items()):
        verdict = judged.get(key)
        if verdict is None:
            missing.append(key)
            continue
        pairs.append((label["verdict"], verdict))
        matrix[label["verdict"]][verdict] += 1
        stats = per_check[label["check"]]
        stats["n"] += 1
        stats["agree"] += int(label["verdict"] == verdict)

    binary = [(("genuine" if a == GENUINE else "not_genuine"),
               ("genuine" if b == GENUINE else "not_genuine")) for a, b in pairs]

    return {
        "suite": suite_dir.name,
        "labels": len(labels),
        "compared": len(pairs),
        "not_yet_adjudicated": missing,
        "binary_kappa": cohens_kappa(binary),
        "three_way_kappa": cohens_kappa(pairs),
        "confusion": {k: dict(v) for k, v in sorted(matrix.items())},
        "per_check_raw_agreement": {k: dict(v) for k, v in sorted(per_check.items())},
    }


def format_score(s: dict[str, Any]) -> str:
    lines = [f"{s['suite']}: {s['compared']} of {s['labels']} hand labels have a judge verdict"]
    if s["not_yet_adjudicated"]:
        lines.append(f"  {len(s['not_yet_adjudicated'])} labelled item(s) not adjudicated yet")
    lines.append("")

    for name, key in (("Binary (genuine vs not)", "binary_kappa"), ("Three-way", "three_way_kappa")):
        k = s[key]
        if not k["n"]:
            continue
        head = (f"{name:26s} raw agreement {k['observed_agreement']:.1%} "
                f"({k['agreements']}/{k['n']})")
        if k["kappa"] is None:
            lines.append(f"{head}   kappa UNDEFINED -- {k.get('undefined_because', '')}")
        else:
            lo, hi = k["ci95"]
            lines.append(f"{head}   kappa {k['kappa']:.2f}  95% CI {lo:.2f} to {hi:.2f}")
    lines.append("")

    lines.append("Confusion (rows = hand label, columns = judge):")
    lines.append(f"  {'':28s}" + "".join(f"{v[:18]:>20s}" for v in VERDICTS))
    for human, row in s["confusion"].items():
        lines.append(f"  {human:28s}" + "".join(f"{row.get(v, 0):>20d}" for v in VERDICTS))
    lines.append("")

    lines.append("Per check (raw agreement only -- too few items each for a kappa):")
    for check, stats in s["per_check_raw_agreement"].items():
        rate = stats["agree"] / stats["n"] if stats["n"] else 0.0
        lines.append(f"  {check:32s} {stats['agree']:2d}/{stats['n']:<2d}  {rate:.0%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hand-label adjudicable failures, blind, to calibrate the adjudicator.")
    parser.add_argument("suite", help="suite run id under evals/runs/, or a path to a run directory")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--labels", type=Path, default=None,
                        help=f"where to read/write labels (default: <suite>/{LABELS_FILENAME})")
    parser.add_argument("--seed", type=int, default=0,
                        help="shuffle order for presentation; recorded in the labels file")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many items")
    parser.add_argument("--full", action="store_true",
                        help="show whole tool results instead of the judge's elided view. Use for "
                             "a second look, knowing that labelling from more evidence than the "
                             "judge received makes disagreement ambiguous")
    parser.add_argument("--list", action="store_true",
                        help="print the labelling queue and exit, without prompting")
    parser.add_argument("--score", action="store_true",
                        help="compare existing labels against the adjudicator's verdicts")
    args = parser.parse_args(argv)

    candidate = Path(args.suite)
    suite_dir = candidate if candidate.is_dir() else args.runs_dir / args.suite
    labels_path = args.labels or suite_dir / LABELS_FILENAME

    if args.score:
        print(format_score(score(suite_dir, labels_path)))
        return 0

    if args.list:
        queue = labelling_queue(suite_dir)
        counts = Counter(item["check"] for item in queue)
        print(f"{len(queue)} distinct (case, check) pairs to label in {suite_dir.name}:")
        for item in queue:
            extra = f"  [{item['observations']} observations]" if item["observations"] > 1 else ""
            print(f"  {item['case_id']:48s} {item['check']}{extra}")
        print("\nby check:")
        for check, count in sorted(counts.items()):
            print(f"  {check:32s} {count}")
        return 0

    if not sys.stdin.isatty():
        print("Labelling needs an interactive terminal; use --list or --score here.",
              file=sys.stderr)
        return 2

    budget = 10 ** 9 if args.full else _DEFAULT_MAX_RESULT_CHARS
    run_session(suite_dir, labels_path=labels_path, budget=budget, seed=args.seed,
                limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
