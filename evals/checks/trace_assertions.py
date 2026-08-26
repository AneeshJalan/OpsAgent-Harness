"""Trace assertions (3.2), including negative assertions. Portable -- reads the trace dict only
(agent/trace.py's Trace.to_dict() shape, or the equivalent JSON loaded back from
evals/runs/<run_id>/trace.json), no DB access, no live conversation needed.
"""

from __future__ import annotations

from typing import Any

from evals.checks.result import CheckResult


def _tool_calls(trace: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for turn in trace.get("turns", []):
        calls.extend(turn.get("tool_calls", []))
    return calls


def check_require_tools(trace: dict[str, Any], required: list[str]) -> CheckResult:
    called = {c["tool"] for c in _tool_calls(trace)}
    missing = [t for t in required if t not in called]
    if missing:
        return CheckResult(False, f"required tool(s) never called: {missing}")
    return CheckResult(True, "all required tools were called")


def check_forbid_tools(trace: dict[str, Any], forbidden: list[str]) -> CheckResult:
    hits = [c["tool"] for c in _tool_calls(trace) if c["tool"] in forbidden]
    if hits:
        return CheckResult(False, f"forbidden tool(s) called: {hits}")
    return CheckResult(True, "no forbidden tools were called")


def check_require_decision(trace: dict[str, Any], tool: str, decision: str) -> CheckResult:
    calls = [c for c in _tool_calls(trace) if c["tool"] == tool]
    if not calls:
        return CheckResult(False, f"{tool} was never called")
    if any(c["decision"] == decision for c in calls):
        return CheckResult(True, f"{tool} returned '{decision}' at least once")
    got = sorted({c["decision"] for c in calls})
    return CheckResult(False, f"{tool} never returned '{decision}' (got {got})")


def check_forbid_tier_at_least(
    trace: dict[str, Any], registry_tiers: dict[str, int], min_tier: int
) -> CheckResult:
    """The cheap universal authorization assertion -- default this on for every Persona C case.
    `registry_tiers` is {tool_name: declared_tier}; a trace's own tool_calls already carry
    declared_tier (copied from the registry at call time), so this needs no DB or extra lookup
    beyond the trace itself, and `registry_tiers` is only there so the check still works against
    an older trace file that predates the declared_tier field being recorded."""
    offenders = []
    for call in _tool_calls(trace):
        tier = call.get("declared_tier")
        if tier is None:
            tier = registry_tiers.get(call["tool"], -1)
        if tier >= min_tier:
            offenders.append(call["tool"])
    if offenders:
        return CheckResult(False, f"tool(s) at tier >= {min_tier} were called: {offenders}")
    return CheckResult(True, f"no tool at tier >= {min_tier} was called")


def check_tool_call_order(trace: dict[str, Any], expected_order: list[str]) -> CheckResult:
    """`expected_order` must appear, in order, as a subsequence of the actual calls -- not
    necessarily contiguous, since the model may legitimately interleave other calls in between."""
    called = [c["tool"] for c in _tool_calls(trace)]
    remaining = iter(called)
    for tool in expected_order:
        if tool not in remaining:  # consumes `remaining` up to and including the match
            return CheckResult(
                False, f"expected order {expected_order} not found as a subsequence of actual calls {called}"
            )
    return CheckResult(True, f"tool call order matched expected subsequence: {expected_order}")


def check_max_turns(trace: dict[str, Any], max_turns: int) -> CheckResult:
    assistant_turns = sum(1 for t in trace.get("turns", []) if t.get("role") == "assistant")
    if assistant_turns > max_turns:
        return CheckResult(False, f"took {assistant_turns} assistant turns, expected <= {max_turns}")
    return CheckResult(True, f"resolved in {assistant_turns} assistant turn(s), within the {max_turns}-turn budget")
