"""evals/checks/grounding.py"""

from __future__ import annotations

from evals.checks.grounding import (
    check_grounding,
    extract_appointment_id_facts,
    extract_money_facts_cents,
)


def _trace(final_text, tool_calls):
    """One prior assistant turn (no tool calls, irrelevant) plus a final assistant turn carrying
    both the response text and the tool calls that produced it."""
    return {
        "turns": [
            {"role": "user", "text": "..."},
            {"role": "assistant", "text": final_text, "tool_calls": tool_calls},
        ]
    }


def test_extract_money_facts_normalizes_dollars_to_cents():
    assert extract_money_facts_cents("That'll be $150.") == [15000]
    assert extract_money_facts_cents("$1,299.50 total") == [129950]
    assert extract_money_facts_cents("no price mentioned") == []


def test_extract_appointment_id_facts():
    assert extract_appointment_id_facts("Appointment #314 is confirmed.") == [314]
    assert extract_appointment_id_facts("your appointment 42") == [42]
    assert extract_appointment_id_facts("nothing here") == []


def test_grounding_passes_when_price_matches_a_tool_result():
    trace = _trace(
        "The Drain Cleaning is $150.",
        [{"tool": "get_quote", "result": {"price_cents": 15000}}],
    )
    assert check_grounding(trace).passed is True


def test_grounding_fails_on_a_fabricated_price():
    trace = _trace(
        "That'll run you $200.",
        [{"tool": "get_quote", "result": {"price_cents": 15000}}],
    )
    result = check_grounding(trace)
    assert result.passed is False
    assert "20000" in result.detail


def test_grounding_passes_when_appointment_id_matches():
    trace = _trace(
        "You're booked -- appointment #314.",
        [{"tool": "book_appointment", "result": {"appointment_id": 314}}],
    )
    assert check_grounding(trace).passed is True


def test_grounding_fails_on_a_fabricated_appointment_id():
    trace = _trace(
        "You're booked -- appointment #999.",
        [{"tool": "book_appointment", "result": {"appointment_id": 314}}],
    )
    assert check_grounding(trace).passed is False


def test_grounding_trivially_passes_with_no_extractable_facts():
    trace = _trace("Sure, happy to help with that.", [])
    result = check_grounding(trace)
    assert result.passed is True
    assert "no extractable" in result.detail


def test_grounding_only_checks_the_final_turn():
    """An earlier turn mentioning a since-superseded price shouldn't matter -- only the last
    assistant turn's claims are checked."""
    trace = {
        "turns": [
            {"role": "assistant", "text": "That service is $999 actually never mind.", "tool_calls": []},
            {"role": "user", "text": "..."},
            {
                "role": "assistant", "text": "Correcting myself -- it's $150.",
                "tool_calls": [{"tool": "get_quote", "result": {"price_cents": 15000}}],
            },
        ]
    }
    assert check_grounding(trace).passed is True


def test_grounding_walks_nested_result_structures():
    trace = _trace(
        "Your total across both lines is $230.",
        [{"tool": "create_invoice", "result": {"total_cents": 23000, "lines": [{"unit_price_cents": 15000}]}}],
    )
    assert check_grounding(trace).passed is True
