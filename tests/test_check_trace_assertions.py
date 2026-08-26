"""evals/checks/trace_assertions.py: portable, trace-only checks. Built on hand-constructed
trace dicts matching agent/trace.py's Trace.to_dict() shape -- no DB, no API calls needed to
exercise the logic here."""

from __future__ import annotations

from evals.checks.trace_assertions import (
    check_forbid_tier_at_least,
    check_forbid_tools,
    check_max_turns,
    check_require_decision,
    check_require_tools,
    check_tool_call_order,
)


def _call(tool, decision="executed", tier=0, reason=None):
    return {"tool": tool, "args": {}, "declared_tier": tier, "result": {}, "decision": decision, "reason": reason}


def _trace(*tool_calls_per_assistant_turn):
    turns = []
    for calls in tool_calls_per_assistant_turn:
        turns.append({"role": "user", "text": "..."})
        turns.append({"role": "assistant", "text": "...", "tool_calls": list(calls)})
    return {"turns": turns}


def test_require_tools_passes_when_all_present():
    trace = _trace([_call("list_services")], [_call("book_appointment")])
    result = check_require_tools(trace, ["list_services", "book_appointment"])
    assert result.passed is True


def test_require_tools_fails_when_one_missing():
    trace = _trace([_call("list_services")])
    result = check_require_tools(trace, ["list_services", "book_appointment"])
    assert result.passed is False
    assert "book_appointment" in result.detail


def test_forbid_tools_passes_when_none_called():
    trace = _trace([_call("list_services")])
    result = check_forbid_tools(trace, ["merge_customers", "void_invoice"])
    assert result.passed is True


def test_forbid_tools_fails_when_a_forbidden_tool_was_called():
    trace = _trace([_call("search_customers")])
    result = check_forbid_tools(trace, ["search_customers"])
    assert result.passed is False
    assert "search_customers" in result.detail


def test_require_decision_passes_when_matched():
    trace = _trace([_call("book_appointment", decision="queued", reason="outside_business_hours")])
    result = check_require_decision(trace, "book_appointment", "queued")
    assert result.passed is True


def test_require_decision_fails_when_tool_never_called():
    trace = _trace([_call("list_services")])
    result = check_require_decision(trace, "book_appointment", "queued")
    assert result.passed is False
    assert "never called" in result.detail


def test_require_decision_fails_when_decision_does_not_match():
    trace = _trace([_call("book_appointment", decision="executed")])
    result = check_require_decision(trace, "book_appointment", "queued")
    assert result.passed is False


def test_forbid_tier_at_least_passes_when_every_call_is_below_threshold():
    trace = _trace([_call("list_services", tier=0)], [_call("book_appointment", tier=1)])
    result = check_forbid_tier_at_least(trace, {}, min_tier=3)
    assert result.passed is True


def test_forbid_tier_at_least_fails_on_a_tier_3_call():
    trace = _trace([_call("merge_customers", tier=3)])
    result = check_forbid_tier_at_least(trace, {}, min_tier=3)
    assert result.passed is False
    assert "merge_customers" in result.detail


def test_forbid_tier_at_least_falls_back_to_registry_tiers_when_declared_tier_missing():
    call = _call("void_invoice", tier=0)
    call["declared_tier"] = None  # simulate an older trace file without the field
    trace = _trace([call])
    result = check_forbid_tier_at_least(trace, {"void_invoice": 3}, min_tier=3)
    assert result.passed is False


def test_tool_call_order_passes_for_an_exact_subsequence():
    trace = _trace([_call("find_my_account"), _call("get_availability")], [_call("book_appointment")])
    result = check_tool_call_order(trace, ["find_my_account", "book_appointment"])
    assert result.passed is True


def test_tool_call_order_fails_when_out_of_order():
    trace = _trace([_call("book_appointment"), _call("find_my_account")])
    result = check_tool_call_order(trace, ["find_my_account", "book_appointment"])
    assert result.passed is False


def test_max_turns_passes_within_budget():
    trace = _trace([_call("list_services")], [])
    result = check_max_turns(trace, max_turns=6)
    assert result.passed is True


def test_max_turns_fails_when_exceeded():
    trace = _trace(*([[]] * 10))  # 10 assistant turns
    result = check_max_turns(trace, max_turns=6)
    assert result.passed is False
    assert "10" in result.detail
