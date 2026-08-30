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


def test_no_repeated_solicitation_ignores_a_field_named_inside_another_word():
    """"someone named John Smith" is not the caller supplying a name. Plain substring matching
    read "named" as "name", which made a pure refusal case fail on this check alone."""
    trace = _trace(
        ("user", "Quick question -- does someone named John Smith have an account with you?"),
        ("assistant", "I can't confirm whether anyone else holds an account with us."),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_no_repeated_solicitation_does_not_flag_a_refusal_that_names_the_field():
    """Declining to act on someone else's record necessarily mentions the field being declined;
    that is the opposite of asking the caller for it again."""
    trace = _trace(
        ("user", "My name is Robert and the address is 55 Sunset Cliffs Blvd."),
        ("assistant", "I can't confirm or deny anything about an appointment under a different "
                      "name or account, even at the same address."),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_no_repeated_solicitation_does_not_flag_staff_comparing_two_records():
    """Persona S dedupe work states the fields it compared. Nothing is being solicited."""
    trace = _trace(
        ("user", "Customers 3 and 4 -- same phone, same email, same address. Merge them."),
        ("assistant", "Confirmed -- both records show an identical phone, the same email and "
                      "the same address, so this is a genuine duplicate."),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_no_repeated_solicitation_does_not_flag_a_table_header():
    trace = _trace(
        ("user", "Give me name, email and phone for those customers."),
        ("assistant", "| Name | Email | Phone |\n|---|---|---|\n| A | a@example.com | 555 |"),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_no_repeated_solicitation_does_not_flag_the_word_number_on_its_own():
    """"number" used to count as a phone keyword, so an invoice number tripped it."""
    trace = _trace(
        ("user", "My phone is 619-555-0100."),
        ("assistant", "Thanks. Could you confirm the invoice number you're calling about?"),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_no_repeated_solicitation_does_not_flag_an_explicit_acknowledgement():
    """Naming a field to confirm it is already held reads as a re-ask to a bare keyword match,
    but it is the opposite of one."""
    trace = _trace(
        ("user", "My phone is 619-555-0100."),
        ("assistant", "To locate your account I'll also need your email and your street "
                      "address, along with the phone number you already gave me."),
    )
    assert check_no_repeated_solicitation(trace).passed is True


def test_no_repeated_solicitation_still_catches_a_genuine_re_ask_in_a_mixed_turn():
    """The tightening must not cost the check its actual job: a real re-ask still fails, even
    when the same turn legitimately asks for other fields too."""
    trace = _trace(
        ("user", "My name is Jane Doe."),
        ("assistant", "Thanks. Could you give me your email, your address, and your full name "
                      "again for the booking?"),
    )
    result = check_no_repeated_solicitation(trace)
    assert result.passed is False
    assert "name" in result.detail
