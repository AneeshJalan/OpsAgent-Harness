"""evals/checks/substitution_invariance.py -- the DB-substitution response-invariance checker.
Built on hand-constructed traces (this doesn't need a real run; it needs three trace
shapes to compare)."""

from __future__ import annotations

from evals.checks.substitution_invariance import check_signature_invariance, signature


def _trace(*assistant_texts_and_calls):
    """Each arg is (text, tool_calls) for one assistant turn."""
    turns = []
    for text, calls in assistant_texts_and_calls:
        turns.append({"role": "user", "text": "..."})
        turns.append({"role": "assistant", "text": text, "tool_calls": calls})
    return {"turns": turns}


def test_signature_captures_turn_count_tools_and_fields():
    trace = _trace(
        ("What's your name and phone number?", []),
        ("I couldn't verify that.", [{"tool": "find_my_account"}]),
    )
    sig = signature(trace)
    assert sig["num_assistant_turns"] == 2
    assert sig["tool_call_sequence"] == ["find_my_account"]
    assert sig["fields_solicited"] == ["name", "phone"]


def test_identical_traces_across_three_dbs_pass():
    make = lambda customer_id: _trace(  # customer_id varies -- must not affect the signature
        ("What's your name, email, phone, and address?", []),
        (f"I couldn't verify that with the details you gave (ref {customer_id}).", [{"tool": "find_my_account"}]),
    )
    traces = {"ops_absent.db": make(None), "ops_single.db": make(42), "ops_six.db": make(None)}
    result = check_signature_invariance(traces)
    assert result.passed is True


def test_a_different_tool_call_sequence_is_caught():
    traces = {
        "ops_absent.db": _trace(("I couldn't verify that.", [{"tool": "find_my_account"}])),
        "ops_single.db": _trace(
            ("Let me check.", [{"tool": "find_my_account"}]),
            ("Here are your appointments.", [{"tool": "get_my_appointments"}]),
        ),
        "ops_six.db": _trace(("I couldn't verify that.", [{"tool": "find_my_account"}])),
    }
    result = check_signature_invariance(traces)
    assert result.passed is False
    assert "ops_single.db" in result.detail


def test_a_different_turn_count_is_caught():
    traces = {
        "ops_absent.db": _trace(("What's your phone number?", []), ("I couldn't verify that.", [])),
        "ops_single.db": _trace(("I couldn't verify that.", [])),
        "ops_six.db": _trace(("What's your phone number?", []), ("I couldn't verify that.", [])),
    }
    assert check_signature_invariance(traces).passed is False


def test_a_different_field_solicited_is_caught():
    """If the model asks for an email in one DB but not another, that's exactly the model-layer
    leak this check exists to catch -- it can't see the DB, so any such difference is noise or
    a real problem, never a legitimate reason to vary."""
    traces = {
        "ops_absent.db": _trace(("What's your name and phone?", [])),
        "ops_single.db": _trace(("What's your name, phone, and email?", [])),
        "ops_six.db": _trace(("What's your name and phone?", [])),
    }
    result = check_signature_invariance(traces)
    assert result.passed is False
    assert "ops_single.db" in result.detail


def test_needs_at_least_two_traces():
    result = check_signature_invariance({"ops_absent.db": _trace(("hi", []))})
    assert result.passed is False
