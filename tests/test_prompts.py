"""agent/prompts.py: the three system prompts must be frozen strings with nothing case- or
run-specific in them -- prompt caching is a prefix match, and any volatile content here silently
invalidates the cache for an entire batch (DAY2 notes: "you just pay 10x" and never notice)."""

from __future__ import annotations

from pathlib import Path

from agent.prompts import SYSTEM_C, SYSTEM_C_POLICY_IN_PROMPT, SYSTEM_S

ALL_PROMPTS = {
    "SYSTEM_C": SYSTEM_C,
    "SYSTEM_S": SYSTEM_S,
    "SYSTEM_C_POLICY_IN_PROMPT": SYSTEM_C_POLICY_IN_PROMPT,
}


def test_prompts_are_nonempty_strings():
    for name, prompt in ALL_PROMPTS.items():
        assert isinstance(prompt, str) and len(prompt) > 200, name


def test_prompts_contain_no_format_placeholders():
    """A stray `{...}` would mean the prompt was meant to be `.format()`-ed with per-run data --
    exactly the volatile-content mistake this module exists to avoid."""
    for name, prompt in ALL_PROMPTS.items():
        assert "{" not in prompt and "}" not in prompt, name


def test_prompts_module_never_imports_datetime_or_random():
    """Source-level guard: nothing in this module should even have the *means* to compute a
    volatile value, let alone use it."""
    source = Path(__file__).resolve().parent.parent.joinpath("agent", "prompts.py").read_text(encoding="utf-8")
    assert "import datetime" not in source
    assert "from datetime" not in source
    assert "import random" not in source
    assert "import uuid" not in source


def test_policy_in_prompt_variant_is_the_base_prompt_plus_more():
    """The ablation adds prose policy guidance on top of SYSTEM_C -- it shouldn't be an
    unrelated rewrite that also changes persona/tone, or the ablation would be confounded."""
    assert SYSTEM_C_POLICY_IN_PROMPT.startswith(SYSTEM_C)
    assert len(SYSTEM_C_POLICY_IN_PROMPT) > len(SYSTEM_C)


def test_prompts_never_instruct_revealing_internal_reason_codes():
    """Mentioning a reason code as an example of what NOT to say is fine (SYSTEM_C does this
    deliberately); actually instructing the model to expose one would not be."""
    for name, prompt in ALL_PROMPTS.items():
        assert "always mention" not in prompt.lower()
        assert "tell the customer the reason code" not in prompt.lower()
