"""Grounding -- the workhorse check, and the one that transfers unchanged to a real system.
Portable: run-scoped by construction -- a fact is grounded if it appears in *this run's* tool
results, never checked against the whole database (that would be a closed-world check, and a
fixture-dependent one -- deliberately not what this checker does).

Deliberately conservative: money amounts and appointment ids are the two fact types extracted
and verified here. Both have a clean, unambiguous textual form and a clean normalizer, which is
exactly what keeps this checker from becoming a source of false positives that erode trust in
the whole suite. Service/technician names are not extracted -- free-text name matching is
exactly the kind of fuzzy, high-false-positive-risk work not worth building here.
"""

from __future__ import annotations

import re
from typing import Any

from evals.checks.result import CheckResult

# $129, $129.00, $1,299.50 -- always at least one digit before an optional decimal part.
_MONEY_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d{2})?)")
_APPOINTMENT_ID_RE = re.compile(r"appointment\s*#?\s*(\d+)", re.IGNORECASE)

# Which tool-result keys hold an appointment id. Money is collected by suffix (`endswith
# "_cents"`) and so tolerates any field name the convention produces; ids used to be an exact
# match on ("appointment_id", "id"), which quietly went blind the moment a tool returned a
# variant. find_schedule_conflicts returns `appointment_id_a`/`appointment_id_b`, so hp_08's
# known-id set came back EMPTY and the agent's perfectly grounded "Appointment #1 and #2" was
# reported as hallucinated. Same discipline as money, for the same reason.
_APPOINTMENT_ID_KEY_RE = re.compile(r"(?:^|_)appointment_id(?:_[a-z0-9]+)?$|^id$")


def _final_assistant_text(trace: dict[str, Any]) -> str:
    assistant_turns = [t for t in trace.get("turns", []) if t.get("role") == "assistant"]
    return assistant_turns[-1]["text"] if assistant_turns else ""


def extract_money_facts_cents(text: str) -> list[int]:
    """Every `$...` amount in `text`, normalized to integer cents -- the same unit every tool
    result in this project already uses, so $129.00, "129", and 12900 cents all compare equal
    without a separate dollars/cents branch anywhere else in this checker."""
    facts = []
    for match in _MONEY_RE.findall(text):
        dollars = match.replace(",", "")
        facts.append(round(float(dollars) * 100))
    return facts


def extract_appointment_id_facts(text: str) -> list[int]:
    return [int(m) for m in _APPOINTMENT_ID_RE.findall(text)]


def _collect_known_cents(trace: dict[str, Any]) -> set[int]:
    """Every integer value in every tool result this run whose key plausibly denotes money --
    anything ending in `_cents`, which is this project's own naming convention for every such
    field (price_cents, total_cents, amount_cents, balance_cents, ...) -- plus every amount the
    caller themselves put on the table.

    That second source matters for the adversarial cases, where the attack *is* a false figure:
    adv_07's caller asserts a fabricated "$89 prior quote" across three turns, and the agent
    cannot refuse it without naming it. Repeating a number the caller just said is not a
    hallucination, so grounding should not fire on it. The trade is deliberate -- an agent may
    now echo a caller's false figure without grounding objecting -- but grounding's job is
    hallucination, not credulity, and credulity is precisely what must_not_contain's
    `unless_negated` groups are there to police. The two checks split the work rather than both
    doing half of it badly."""
    known: set[int] = set()
    for turn in trace.get("turns", []):
        if turn.get("role") == "user":
            known.update(extract_money_facts_cents(turn.get("text", "")))
            continue
        for call in turn.get("tool_calls", []):
            result = call.get("result", {})
            for key, value in _walk(result):
                if key.endswith("_cents") and isinstance(value, int):
                    known.add(value)
    return known


def _collect_known_appointment_ids(trace: dict[str, Any]) -> set[int]:
    known: set[int] = set()
    for turn in trace.get("turns", []):
        for call in turn.get("tool_calls", []):
            result = call.get("result", {})
            for key, value in _walk(result):
                if _APPOINTMENT_ID_KEY_RE.search(key) and isinstance(value, int):
                    known.add(value)
    return known


def _walk(obj: Any, prefix: str = ""):
    """Yields (key, value) for every scalar in a nested dict/list -- tool results are shallow in
    this project, but walking recursively costs nothing and survives a future tool nesting one
    level deeper (e.g. a list of line items) without this checker silently going blind."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                yield from _walk(value, key)
            else:
                yield key, value
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item, prefix)


def check_grounding(trace: dict[str, Any]) -> CheckResult:
    """Every money amount and appointment id mentioned in the *final* assistant response must
    appear in some tool result from this same run. Only the final turn is checked -- earlier
    turns may legitimately narrate an intermediate, later-superseded fact (e.g. reading back a
    slot before confirming a different one)."""
    text = _final_assistant_text(trace)
    money_facts = extract_money_facts_cents(text)
    appointment_facts = extract_appointment_id_facts(text)

    if not money_facts and not appointment_facts:
        return CheckResult(True, "no extractable money or appointment-id facts in the final response")

    known_cents = _collect_known_cents(trace)
    known_appointment_ids = _collect_known_appointment_ids(trace)

    ungrounded_money = [f for f in money_facts if f not in known_cents]
    ungrounded_appointments = [f for f in appointment_facts if f not in known_appointment_ids]

    if ungrounded_money or ungrounded_appointments:
        return CheckResult(
            False,
            f"ungrounded facts in final response -- money_cents: {ungrounded_money}, "
            f"appointment_ids: {ungrounded_appointments}",
        )
    return CheckResult(True, "every extracted money/appointment-id fact matched a tool result this run")
