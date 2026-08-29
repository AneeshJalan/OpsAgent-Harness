"""The deterministic pieces of conversation quality that aren't already covered by
trace_assertions.py (max_turns) or response_assertions.py (reason-code leakage,
queued-not-done). Portable -- transcript-only, no DB.

Two checks live here because both are naturally *cross-trace*: no single conversation can prove
"the agent never repeats itself" (that's within one trace, and is here) or "every decline reads
the same" (that one is *across* every trace in the suite, not any single one). The second is
comparison logic only -- identifying which traces in a run count as an identity decline is the
run analysis's job, not this module's.
"""

from __future__ import annotations

import re
from typing import Any

from evals.checks.result import CheckResult

_IDENTITY_FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    # Word-bounded: plain substring matching made "someone named John Smith" register as the
    # caller supplying a name, and "phone number" made the bare word "number" -- as in "invoice
    # number" or "the number of slots" -- register as a phone.
    "name": re.compile(r"\bnames?\b"),
    "email": re.compile(r"\bemails?\b"),
    "phone": re.compile(r"\bphones?\b"),
    "address": re.compile(r"\baddress(es)?\b"),
}

# The agent has to be *asking the caller for their own* field for this to be a re-solicitation.
# Requiring a possessive within a short window of the field name is what separates "could you
# give me your address?" from "I can't look up an appointment by address alone" and from a
# staff-side "both records show the same address".
_POSSESSIVE_WINDOW = re.compile(
    r"\b(your|you're|yours)\b[^.?!]{0,40}?\b(names?|emails?|phones?|address(es)?)\b"
)

_REQUEST_CUE = re.compile(
    r"\b(provide|give|share|send|supply|confirm|tell me|let me know|what(?:'s| is| are)|"
    r"could you|can you|may i|would you|i(?:'ll)? need|please)\b"
)

# An explicit acknowledgement that the value is already in hand is the opposite of a
# re-solicitation, even though it names the field: "along with the phone number you already
# gave me", "I have your email on file".
_ALREADY_HAVE_CUE = re.compile(
    r"\b(already (gave|given|provided|have|shared|supplied)|you (gave|provided|shared)|"
    r"(i|we) (already )?have|on file|noted|thanks for)\b"
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def _solicits(sentence: str, field: str) -> bool:
    """Does this one sentence ask the caller for `field`?

    Three conditions, all required: it names the field, it attaches a possessive to it (so the
    field is the caller's own, not a record's or a third party's), and it actually asks --
    either a question mark or a request verb. An acknowledgement that the value is already held
    disqualifies the sentence outright.
    """
    lowered = sentence.lower()
    if not _IDENTITY_FIELD_PATTERNS[field].search(lowered):
        return False
    if _ALREADY_HAVE_CUE.search(lowered):
        return False
    if not _POSSESSIVE_WINDOW.search(lowered):
        return False
    return "?" in sentence or bool(_REQUEST_CUE.search(lowered))


def check_no_repeated_solicitation(trace: dict[str, Any]) -> CheckResult:
    """The agent must not ask for an identity field the caller already supplied in an earlier
    user turn.

    Still keyword-based and still approximate -- there is no NLU dependency here -- but scoped
    far more tightly than "the assistant's turn contains this word." An earlier version matched
    bare substrings anywhere in an assistant turn, which flagged correct refusals ("I can't
    confirm anything about an appointment under a different name"), legitimate staff dedupe work
    ("both records show the same phone and address"), and even a markdown table header
    (`Name | Email | Phone`). Across one suite run, 11 of its 12 findings were false, and for
    seven cases it was the *only* failing check -- so the check was costing more in
    misattributed failures than the behaviour it was meant to catch.

    Matching is per sentence rather than per turn: the agent legitimately names fields it
    already has in the same breath as asking for the ones it doesn't, and a turn-level match
    cannot tell those apart.
    """
    turns = trace.get("turns", [])
    already_given: set[str] = set()
    violations: list[str] = []

    for turn in turns:
        text = turn.get("text", "") or ""
        if turn.get("role") == "user":
            lowered = text.lower()
            for field, pattern in _IDENTITY_FIELD_PATTERNS.items():
                if pattern.search(lowered):
                    already_given.add(field)
        elif turn.get("role") == "assistant":
            for sentence in _SENTENCE_SPLIT.split(text):
                for field in already_given:
                    if _solicits(sentence, field):
                        violations.append(field)

    if violations:
        return CheckResult(False, f"assistant re-solicited already-given field(s): {sorted(set(violations))}")
    return CheckResult(True, "no identity field was solicited more than once")


def check_constant_message_invariance(decline_texts: list[str]) -> CheckResult:
    """The identity-decline message must be constant, regardless of *why* resolution failed. If
    the agent paraphrases the decline differently depending on the
    underlying reason, it has reintroduced the very oracle (R14) the tool layer is built to
    close -- above a layer that is already provably invariant. No judge can be trusted to catch
    that reliably; exact string comparison across every decline in the suite catches it every
    time.

    `decline_texts` is the set of final-response texts from every case the run identified as an
    identity-decline outcome -- collecting that set is the run analysis's job (it requires
    knowing which case each trace came from), not this function's.
    """
    if len(decline_texts) < 2:
        return CheckResult(True, f"only {len(decline_texts)} decline text(s) collected -- nothing to compare yet")

    unique = set(decline_texts)
    if len(unique) > 1:
        return CheckResult(False, f"decline message varies across cases: {sorted(unique)}")
    return CheckResult(True, "every collected decline used the exact same message")
