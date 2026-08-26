"""The remaining deterministic pieces of conversation quality (3.8) that aren't already covered
by trace_assertions.py (max_turns) or response_assertions.py (reason-code leakage,
queued-not-done). Portable -- transcript-only, no DB.

Two checks live here because both are naturally *cross-trace*: no single conversation can prove
"the agent never repeats itself" (that's within one trace, and is here) or "every decline reads
the same" (that one is *across* every trace in the suite, not any single one). The second is
comparison logic only -- identifying which traces in a run count as an identity decline is the
run analysis's job, not this module's.
"""

from __future__ import annotations

from typing import Any

from evals.checks.result import CheckResult

_IDENTITY_FIELD_KEYWORDS: dict[str, list[str]] = {
    "name": ["name"],
    "email": ["email"],
    "phone": ["phone", "number"],
    "address": ["address"],
}


def check_no_repeated_solicitation(trace: dict[str, Any]) -> CheckResult:
    """The agent must not ask for an identity field the caller already supplied in an earlier
    user turn. Detected the same keyword way as the substitution-invariance signature: does an
    *assistant* turn's text ask about a field, after a *user* turn already mentioned it?

    Deliberately approximate -- this flags "the agent asked about X again after the user already
    said X," using field-name keywords, not a full parse of whether the user's mention actually
    supplied a usable value. False positives are possible (an assistant turn using the word
    "email" for another reason after the user mentioned email) but rare enough in this domain's
    conversations to keep the check useful without needing an NLU dependency.
    """
    turns = trace.get("turns", [])
    already_given: set[str] = set()
    violations: list[str] = []

    for turn in turns:
        text_lower = turn.get("text", "").lower()
        if turn.get("role") == "user":
            for field, keywords in _IDENTITY_FIELD_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    already_given.add(field)
        elif turn.get("role") == "assistant":
            for field, keywords in _IDENTITY_FIELD_KEYWORDS.items():
                if field in already_given and any(kw in text_lower for kw in keywords):
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
