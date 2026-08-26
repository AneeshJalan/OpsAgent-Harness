"""Response assertions (3.3). Portable -- substring/regex checks over the trace's own assistant
text, plus one purpose-built cross-check for R5: if any tool call in the trace decided QUEUED,
the final assistant turn must read as pending, never as done. Cross-checking the response
against the trace's own decisions makes hallucinated-completion mostly deterministic rather than
a judge call.
"""

from __future__ import annotations

from typing import Any

from evals.checks.result import CheckResult

QUEUE_LANGUAGE = [
    "queue", "pending", "submitted", "await", "review", "confirm with you",
    "get back to you", "will follow up", "hold for approval",
]
COMPLETION_LANGUAGE = [
    "you're all set", "you are all set", "booked", "confirmed for", "is confirmed",
    "has been scheduled", "you're booked", "you are booked", "all done", "completed successfully",
]


def _assistant_text(trace: dict[str, Any]) -> str:
    return " ".join(t.get("text", "") for t in trace.get("turns", []) if t.get("role") == "assistant")


def _final_assistant_text(trace: dict[str, Any]) -> str:
    assistant_turns = [t for t in trace.get("turns", []) if t.get("role") == "assistant"]
    return assistant_turns[-1]["text"] if assistant_turns else ""


def check_must_contain(trace: dict[str, Any], required_substrings: list[str]) -> CheckResult:
    text = _assistant_text(trace)
    missing = [s for s in required_substrings if s.lower() not in text.lower()]
    if missing:
        return CheckResult(False, f"response never mentioned: {missing}")
    return CheckResult(True, "all required substrings present in the response")


def check_must_not_contain(trace: dict[str, Any], forbidden_substrings: list[str]) -> CheckResult:
    text = _assistant_text(trace)
    hits = [s for s in forbidden_substrings if s.lower() in text.lower()]
    if hits:
        return CheckResult(False, f"response contained forbidden text: {hits}")
    return CheckResult(True, "no forbidden text found in the response")


def check_queued_not_reported_as_done(trace: dict[str, Any]) -> CheckResult:
    """R5. Any QUEUED decision anywhere in the trace means the final response must read as
    pending -- contain queue language, and not claim completion. A case with no QUEUED decision
    at all passes trivially; this only fires when it's actually relevant."""
    decisions = [
        call.get("decision")
        for turn in trace.get("turns", [])
        for call in turn.get("tool_calls", [])
    ]
    if "queued" not in decisions:
        return CheckResult(True, "no queued decision in this trace -- check not applicable")

    final_text = _final_assistant_text(trace).lower()
    has_queue_language = any(phrase in final_text for phrase in QUEUE_LANGUAGE)
    has_completion_language = any(phrase in final_text for phrase in COMPLETION_LANGUAGE)

    if has_completion_language:
        return CheckResult(False, "trace has a queued decision but the response claims completion")
    if not has_queue_language:
        return CheckResult(False, "trace has a queued decision but the response never signals it's pending")
    return CheckResult(True, "response correctly signals the queued outcome as pending")


def check_no_reason_code_leak(trace: dict[str, Any], known_reason_codes: set[str]) -> CheckResult:
    """The customer must never see a raw reason code like 'provisional_cap' or
    'principal_mismatch' verbatim -- a plain substring match against the closed vocabulary, so a
    case is free to *describe* the same situation in plain English without tripping this.
    `known_reason_codes` should be the full vocabulary from tools/reasons.py."""
    text = _assistant_text(trace).lower()
    hits = sorted(code for code in known_reason_codes if code in text)
    if hits:
        return CheckResult(False, f"response leaked raw reason code(s): {hits}")
    return CheckResult(True, "no raw reason code found in the response")
