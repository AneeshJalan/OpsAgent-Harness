"""evals/checks/conversation_quality.py"""

from __future__ import annotations

from evals.checks.conversation_quality import (
    check_constant_message_invariance,
    check_no_repeated_solicitation,
)


def _trace(*role_text_pairs):
    return {"turns": [{"role": role, "text": text} for role, text in role_text_pairs]}


def test_no_repeated_solicitation_passes_when_each_field_asked_once():
    trace = _trace(
        ("assistant", "What's your name and phone number?"),
        ("user", "Jane Doe, 619-555-0100."),
        ("assistant", "Great, and your email and address?"),
        ("user", "jane@example.com, 1 Main St."),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_no_repeated_solicitation_fails_when_a_given_field_is_asked_again():
    trace = _trace(
        ("assistant", "What's your phone number?"),
        ("user", "My phone is 619-555-0100."),
        ("assistant", "Sorry, could you give me your phone number again?"),
    )
    result = check_no_repeated_solicitation(trace)
    assert result.passed is False
    assert "phone" in result.detail


def test_no_repeated_solicitation_does_not_flag_fields_never_given():
    """Asking for the same missing field twice because the user never answered isn't a repeat
    solicitation of a *given* field -- the check only cares about fields already supplied."""
    trace = _trace(
        ("assistant", "What's your email?"),
        ("user", "Sorry, what was the question?"),
        ("assistant", "No worries -- what's your email?"),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_constant_message_invariance_passes_with_identical_messages():
    result = check_constant_message_invariance([
        "I wasn't able to verify your account with those details.",
        "I wasn't able to verify your account with those details.",
        "I wasn't able to verify your account with those details.",
    ])
    assert result.passed is True


def test_constant_message_invariance_fails_on_a_paraphrase():
    result = check_constant_message_invariance([
        "I wasn't able to verify your account with those details.",
        "Hmm, that doesn't match anyone in our system.",
    ])
    assert result.passed is False


def test_constant_message_invariance_trivially_passes_with_fewer_than_two_messages():
    """Just the degenerate-input edge case -- the real ≥2-message invariance logic is exercised
    by the two tests above (identical messages pass, a paraphrase fails)."""
    assert check_constant_message_invariance([]).passed is True
    assert check_constant_message_invariance(["only one"]).passed is True
