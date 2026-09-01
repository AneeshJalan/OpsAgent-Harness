"""evals/label_adjudication.py -- blind hand-labelling, and the agreement it makes measurable.

The interactive loop is driven by injected read/write callables, so the whole session is testable
without a terminal.

The property most worth protecting is blindness: nothing the labeller is shown may reveal what the
adjudicator said, or the labels stop being able to measure it. There is a test for that, and it
adjudicates the case first so a leak would actually have something to leak.
"""

from __future__ import annotations

import json

import pytest

from evals.adjudicate import adjudicate_case
from evals.adjudication import CHECKER_FALSE_POSITIVE, GENUINE, INSUFFICIENT_EVIDENCE
from evals.label_adjudication import (
    cohens_kappa,
    format_score,
    label_key,
    labelling_queue,
    load_labels,
    main,
    render_item,
    run_session,
    score,
)
from tests.test_adjudicate import ScriptedJudge, verdict_response, write_case


def build_suite(tmp_path):
    suite = tmp_path / "suite-1"
    suite.mkdir(parents=True)
    write_case(suite, "hap_01", passed=True)
    write_case(suite, "adv_07", passed=False, failing=["must_not_contain"])
    write_case(suite, "adv_08", passed=False, failing=["no_pii", "grounding"])
    write_case(suite, "auth_03", passed=False, failing=["require_tools"])
    return suite


class Replies:
    """Feeds scripted keystrokes to the session and captures everything written."""

    def __init__(self, answers):
        self._answers = list(answers)
        self.out = []

    def read(self, _prompt=""):
        return self._answers.pop(0) if self._answers else "q"

    def write(self, text):
        self.out.append(str(text))

    @property
    def text(self):
        return "\n".join(self.out)


# --- what gets queued ---------------------------------------------------------------------------


def test_only_adjudicable_failures_are_queued(tmp_path):
    queue = labelling_queue(build_suite(tmp_path))
    assert [(i["case_id"], i["check"]) for i in queue] == [
        ("adv_07", "must_not_contain"), ("adv_08", "grounding"), ("adv_08", "no_pii")]
    # hap_01 passed; auth_03 fails only an exact check no judge may overturn.


def test_replicates_of_one_pair_collapse_to_a_single_item(tmp_path):
    # Two observations of the same case failing the same check are not independent items, and
    # counting both would make every interval look tighter than it is.
    suite = tmp_path / "suite-1"
    suite.mkdir()
    write_case(suite, "adv_07", passed=False, failing=["must_not_contain"])
    write_case(suite, "adv_07", passed=False, failing=["must_not_contain"])

    queue = labelling_queue(suite)
    assert len(queue) == 1
    assert queue[0]["observations"] == 2
    # Still tied to one real transcript -- there is no average of two conversations.
    assert queue[0]["run_id"] is not None


def test_a_harness_error_is_never_queued(tmp_path):
    suite = tmp_path / "suite-1"
    suite.mkdir()
    write_case(suite, "err_01", passed=None, outcome="harness_error", failing=["no_pii"])
    assert labelling_queue(suite) == []


# --- blindness ----------------------------------------------------------------------------------


def test_the_labeller_is_never_shown_the_judges_verdict(tmp_path):
    # Adjudicate first, with distinctive strings, so a leak would have something to leak. The
    # verdict NAMES legitimately appear in the menu of choices, so the tell is the judge's own
    # evidence and rationale text -- which exists nowhere but adjudication.json and result.json.
    suite = build_suite(tmp_path)
    case_dir = next(d for d in sorted(suite.iterdir()) if d.name.startswith("adv_07"))
    adjudicate_case(case_dir, client=ScriptedJudge([verdict_response(
        CHECKER_FALSE_POSITIVE, evidence="JUDGE-EVIDENCE-MARKER",
        rationale="JUDGE-RATIONALE-MARKER")]))

    item = next(i for i in labelling_queue(suite) if i["case_id"] == "adv_07")
    rendered = render_item(item, 1, 1, 4000)

    assert "JUDGE-EVIDENCE-MARKER" not in rendered
    assert "JUDGE-RATIONALE-MARKER" not in rendered
    assert "adjudicated_by" not in rendered
    assert "passed_adjudicated" not in rendered
    # What it SHOULD show: the claim under audit and the checker's own detail.
    assert "must_not_contain" in rendered
    assert "forbids the assistant from stating" in rendered


def test_the_queue_reads_nothing_the_judge_wrote(tmp_path):
    # Structural version of the same guarantee: labelling a suite that has been fully adjudicated
    # produces byte-identical items to labelling one that has not.
    before = labelling_queue(build_suite(tmp_path / "a"))
    suite = build_suite(tmp_path / "b")
    for case_dir in sorted(suite.iterdir()):
        adjudicate_case(case_dir, client=ScriptedJudge(
            [verdict_response(CHECKER_FALSE_POSITIVE)] * 2))

    strip = lambda q: [{k: v for k, v in i.items() if k not in ("run_id", "case_dir")} for i in q]
    assert strip(labelling_queue(suite)) == strip(before)


def test_the_labeller_is_not_told_the_cases_other_failures(tmp_path):
    suite = tmp_path / "suite-1"
    suite.mkdir()
    write_case(suite, "adv_09", passed=False, failing=["no_pii", "require_tools"])
    rendered = render_item(labelling_queue(suite)[0], 1, 1, 4000)

    assert "require_tools" not in rendered
    assert "no_pii" in rendered


# --- the session ----------------------------------------------------------------------------------


def test_a_label_is_saved_immediately_not_at_the_end(tmp_path):
    # An hour of human attention must not be lost to a closed terminal.
    suite = build_suite(tmp_path)
    labels = tmp_path / "labels.json"
    replies = Replies(["1", "q"])

    run_session(suite, labels_path=labels, read=replies.read, write=replies.write)

    saved = load_labels(labels)["labels"]
    assert len(saved) == 1
    assert next(iter(saved.values()))["verdict"] == GENUINE


def test_every_verdict_key_records_the_right_label(tmp_path):
    suite = build_suite(tmp_path)
    labels = tmp_path / "labels.json"
    replies = Replies(["1", "2", "3"])

    data = run_session(suite, labels_path=labels, read=replies.read, write=replies.write)
    assert sorted(v["verdict"] for v in data["labels"].values()) == sorted(
        [GENUINE, CHECKER_FALSE_POSITIVE, INSUFFICIENT_EVIDENCE])


def test_skip_records_nothing_and_is_not_an_undetermined_verdict(tmp_path):
    # Deferring an item and judging that the evidence cannot settle it are different acts.
    suite = build_suite(tmp_path)
    labels = tmp_path / "labels.json"
    replies = Replies(["s", "s", "s"])

    assert run_session(suite, labels_path=labels, read=replies.read, write=replies.write)["labels"] == {}


def test_a_note_is_attached_to_the_next_verdict(tmp_path):
    suite = build_suite(tmp_path)
    labels = tmp_path / "labels.json"
    replies = Replies(["n", "forbidden string sits inside a refusal", "2", "q"])

    data = run_session(suite, labels_path=labels, read=replies.read, write=replies.write)
    entry = next(iter(data["labels"].values()))
    assert entry["note"] == "forbidden string sits inside a refusal"
    assert entry["verdict"] == CHECKER_FALSE_POSITIVE


def test_an_unrecognised_key_reprompts_rather_than_recording(tmp_path):
    suite = build_suite(tmp_path)
    labels = tmp_path / "labels.json"
    replies = Replies(["x", "9", "1", "q"])

    data = run_session(suite, labels_path=labels, read=replies.read, write=replies.write)
    assert len(data["labels"]) == 1
    assert "unrecognised" in replies.text


def test_the_session_resumes_where_it_stopped(tmp_path):
    suite = build_suite(tmp_path)
    labels = tmp_path / "labels.json"

    first = Replies(["1", "q"])
    run_session(suite, labels_path=labels, read=first.read, write=first.write)
    second = Replies(["2", "q"])
    data = run_session(suite, labels_path=labels, read=second.read, write=second.write)

    assert len(data["labels"]) == 2
    assert "2 left to label" in second.text


def test_the_shuffle_seed_is_recorded_so_the_order_is_reproducible(tmp_path):
    suite = build_suite(tmp_path)
    replies = Replies(["q"])
    data = run_session(suite, labels_path=tmp_path / "l.json", seed=7,
                       read=replies.read, write=replies.write)
    assert data["shuffle_seed"] == 7
    assert data["max_result_chars"] == 4000


# --- kappa ------------------------------------------------------------------------------------------


def test_perfect_agreement_on_a_mixed_sample_is_kappa_one():
    k = cohens_kappa([("a", "a")] * 5 + [("b", "b")] * 5)
    assert k["observed_agreement"] == 1.0
    assert k["kappa"] == pytest.approx(1.0)


def test_kappa_is_undefined_rather_than_zero_when_chance_agreement_is_total():
    # Both raters used one category. The formula is 0/0 there, and reporting 0.0 would say "no
    # better than chance" about data showing perfect agreement.
    k = cohens_kappa([("genuine", "genuine")] * 8)
    assert k["observed_agreement"] == 1.0
    assert k["kappa"] is None
    assert "single category" in k["undefined_because"]


def test_high_raw_agreement_can_still_be_a_modest_kappa():
    # The prevalence trap the binary headline exists to expose: 90% agreement, one rare class.
    pairs = [("genuine", "genuine")] * 17 + [("other", "genuine"), ("genuine", "other")]
    k = cohens_kappa(pairs)
    assert k["observed_agreement"] == pytest.approx(17 / 19)
    assert k["kappa"] < 0.5


def test_kappa_on_an_empty_sample_is_none_not_a_crash():
    assert cohens_kappa([])["kappa"] is None


# --- scoring against the judge ------------------------------------------------------------------


def label_everything(suite, labels_path, verdicts):
    data = {"suite": suite.name, "labels": {}}
    for item in labelling_queue(suite):
        data["labels"][label_key(item)] = {
            "case_id": item["case_id"], "run_id": item["run_id"], "check": item["check"],
            "verdict": verdicts[item["check"]], "note": "", "labelled_at": "now",
        }
    labels_path.write_text(json.dumps(data), encoding="utf-8")


def test_score_compares_labels_against_the_judge_on_the_same_run(tmp_path):
    suite = build_suite(tmp_path)
    for case_dir in sorted(suite.iterdir()):
        result = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
        scripted = {
            "adv_07": [verdict_response(CHECKER_FALSE_POSITIVE)],
            "adv_08": [verdict_response(GENUINE), verdict_response(GENUINE)],
        }.get(result["case_id"], [])
        adjudicate_case(case_dir, client=ScriptedJudge(scripted))

    labels = tmp_path / "labels.json"
    label_everything(suite, labels, {"must_not_contain": CHECKER_FALSE_POSITIVE,
                                     "no_pii": GENUINE, "grounding": INSUFFICIENT_EVIDENCE})

    s = score(suite, labels)
    assert s["compared"] == 3
    assert s["three_way_kappa"]["agreements"] == 2       # grounding disagrees
    assert s["per_check_raw_agreement"]["grounding"] == {"n": 1, "agree": 0}
    assert s["confusion"][INSUFFICIENT_EVIDENCE][GENUINE] == 1
    assert format_score(s)


def test_score_names_labels_the_judge_has_not_reached_yet(tmp_path):
    suite = build_suite(tmp_path)
    labels = tmp_path / "labels.json"
    label_everything(suite, labels, {"must_not_contain": GENUINE, "no_pii": GENUINE,
                                     "grounding": GENUINE})

    s = score(suite, labels)
    assert s["compared"] == 0
    assert len(s["not_yet_adjudicated"]) == 3
    assert "not adjudicated yet" in format_score(s)


def test_score_refuses_when_there_are_no_labels(tmp_path):
    with pytest.raises(SystemExit):
        score(build_suite(tmp_path), tmp_path / "missing.json")


# --- the CLI ------------------------------------------------------------------------------------


def test_list_prints_the_queue_without_prompting(tmp_path, capsys):
    assert main([str(build_suite(tmp_path)), "--list"]) == 0
    out = capsys.readouterr().out
    assert "3 distinct (case, check) pairs" in out
    assert "adv_07" in out and "must_not_contain" in out
    assert "auth_03" not in out      # exact-check-only failures are not labellable


def test_labelling_refuses_a_non_interactive_terminal(tmp_path, capsys):
    # pytest captures stdin, so this is the path a piped or CI invocation takes.
    assert main([str(build_suite(tmp_path))]) == 2
    assert "interactive terminal" in capsys.readouterr().err
