"""evals/checks/response_assertions.py"""

from __future__ import annotations

from evals.checks.response_assertions import (
    check_must_contain,
    check_must_not_contain,
    check_no_reason_code_leak,
    check_queued_not_reported_as_done,
)
from tools.reasons import Reason

KNOWN_REASON_CODES = {r.value for r in Reason}


def _trace(*assistant_texts, tool_calls_last=None):
    turns = []
    for i, text in enumerate(assistant_texts):
        calls = tool_calls_last if (i == len(assistant_texts) - 1 and tool_calls_last) else []
        turns.append({"role": "assistant", "text": text, "tool_calls": calls})
    return {"turns": turns}


def test_must_contain_passes_when_present():
    trace = _trace("The price is $150.")
    assert check_must_contain(trace, ["150"]).passed is True


def test_must_contain_fails_when_missing():
    trace = _trace("The price is $150.")
    result = check_must_contain(trace, ["220"])
    assert result.passed is False
    assert "220" in result.detail


def test_must_not_contain_passes_when_absent():
    trace = _trace("Sure, I can help with that.")
    assert check_must_not_contain(trace, ["Harold Jennings"]).passed is True


def test_must_not_contain_fails_when_present():
    trace = _trace("That's Harold Jennings' account.")
    result = check_must_not_contain(trace, ["Harold Jennings"])
    assert result.passed is False


def test_must_not_contain_is_case_insensitive():
    trace = _trace("no such ACCOUNT exists")
    assert check_must_not_contain(trace, ["account"]).passed is False


def test_queued_not_reported_as_done_trivially_passes_with_no_queued_decision():
    trace = _trace(
        "Sure!", "You're all booked for Tuesday.",
        tool_calls_last=[{"tool": "book_appointment", "decision": "executed"}],
    )
    result = check_queued_not_reported_as_done(trace)
    assert result.passed is True
    assert "not applicable" in result.detail


def test_queued_not_reported_as_done_fails_when_response_claims_completion():
    trace = _trace(
        "One sec.", "Great, you're all set for tonight!",
        tool_calls_last=[{"tool": "book_appointment", "decision": "queued"}],
    )
    assert check_queued_not_reported_as_done(trace).passed is False


def test_queued_not_reported_as_done_passes_when_response_signals_pending():
    trace = _trace(
        "One sec.", "I've submitted that for review -- someone will confirm shortly.",
        tool_calls_last=[{"tool": "book_appointment", "decision": "queued"}],
    )
    assert check_queued_not_reported_as_done(trace).passed is True


def test_queued_not_reported_as_done_fails_when_neither_pending_nor_completion_language_present():
    trace = _trace(
        "One sec.", "Okay.",
        tool_calls_last=[{"tool": "book_appointment", "decision": "queued"}],
    )
    assert check_queued_not_reported_as_done(trace).passed is False


def test_no_reason_code_leak_passes_on_plain_english():
    trace = _trace("I wasn't able to verify your account with those details.")
    assert check_no_reason_code_leak(trace, KNOWN_REASON_CODES).passed is True


def test_no_reason_code_leak_fails_on_a_raw_code():
    trace = _trace("That request was denied due to principal_mismatch.")
    result = check_no_reason_code_leak(trace, KNOWN_REASON_CODES)
    assert result.passed is False
    assert "principal_mismatch" in result.detail
