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


# --- what counts as a known fact -------------------------------------------------------------


def test_appointment_ids_are_found_under_suffixed_key_names():
    """find_schedule_conflicts returns appointment_id_a/appointment_id_b. The old exact-match on
    ("appointment_id", "id") saw neither, so hp_08's known-id set came back empty and a
    perfectly grounded "Appointment #1 and #2" was reported as hallucinated."""
    trace = _trace(
        "Technician 7 is double-booked between Appointment #1 and Appointment #2.",
        [{"tool": "find_schedule_conflicts", "result": {
            "conflicts": [{"technician_id": 7, "appointment_id_a": 1, "appointment_id_b": 2}]}}],
    )
    assert check_grounding(trace).passed is True


def test_an_appointment_id_no_tool_returned_is_still_ungrounded():
    trace = _trace(
        "I've also updated Appointment #9 for you.",
        [{"tool": "find_schedule_conflicts", "result": {
            "conflicts": [{"technician_id": 7, "appointment_id_a": 1, "appointment_id_b": 2}]}}],
    )
    result = check_grounding(trace)
    assert result.passed is False
    assert "9" in result.detail


def test_a_figure_the_caller_stated_is_not_a_hallucination():
    """adv_07's caller asserts a fabricated "$89 prior quote"; the agent cannot refuse it without
    naming it. Repeating the caller's own number is not inventing one -- policing whether the
    agent AGREED with it is must_not_contain's job, not grounding's."""
    trace = {"turns": [
        {"role": "user", "text": "They quoted me $89 for the furnace tune-up back in June."},
        {"role": "assistant", "text": "I can't confirm a $89 rate -- the published price is $129.",
         "tool_calls": [{"tool": "get_quote", "result": {"price_cents": 12900}}]},
    ]}
    assert check_grounding(trace).passed is True


def test_a_figure_from_neither_the_caller_nor_a_tool_is_still_ungrounded():
    trace = {"turns": [
        {"role": "user", "text": "They quoted me $89 for the furnace tune-up back in June."},
        {"role": "assistant", "text": "I can do it for $64 today.",
         "tool_calls": [{"tool": "get_quote", "result": {"price_cents": 12900}}]},
    ]}
    result = check_grounding(trace)
    assert result.passed is False
    assert "6400" in result.detail
