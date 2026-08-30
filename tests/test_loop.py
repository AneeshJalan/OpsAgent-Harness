"""agent/loop.py: the manual tool-use agent loop. Every test here mocks the Anthropic client --
no network access, no API key, no real model call anywhere in this file. What's under test is
the loop's own control flow: it calls dispatch() and never a registry function directly, batches
parallel tool results into one message, plays scripted user turns in order, enforces max_turns,
strips the principal/run_id boundary defensively, and degrades to a harness_error outcome on the
SDK's typed exceptions instead of crashing.
"""

from __future__ import annotations

import anthropic
import httpx2
import pytest

from agent.loop import run_agent
from fakes import FakeAnthropicClient, FakeMessage, FakeTextBlock, FakeToolUseBlock, FakeUsage
from tools.dispatcher import ToolSpec
from tools.principal import Principal

FAKE_REGISTRY = {"echo": ToolSpec(fn=lambda **kw: {"decision": "executed"}, tier=0)}
DESCRIPTIONS = {"echo": "Echoes back its arguments."}
CUSTOMER = Principal(type="customer", id=1)


def _end_turn(text: str) -> FakeMessage:
    return FakeMessage(content=[FakeTextBlock(text=text)], stop_reason="end_turn")


def test_happy_path_no_tool_calls():
    client = FakeAnthropicClient([_end_turn("Sure, here's the info you asked for.")])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["What are your hours?"], descriptions=DESCRIPTIONS,
        run_id="run-1", client=client,
    )
    assert trace.outcome == "ok"
    assert [t.role for t in trace.turns] == ["user", "assistant"]
    assert trace.turns[1].text == "Sure, here's the info you asked for."
    assert trace.turns[1].tool_calls == []
    assert trace.hit_turn_cap is False
    assert len(client.calls) == 1


def test_tool_call_goes_through_dispatch_not_the_registry_function_directly(monkeypatch):
    calls = []

    def fake_dispatch(registry, tool_name, principal, *, run_id=None, **kwargs):
        calls.append((registry, tool_name, principal, run_id, kwargs))
        return {"decision": "executed", "reason": None, "entity_ref": "customer:1"}

    monkeypatch.setattr("agent.loop.dispatch", fake_dispatch)

    client = FakeAnthropicClient([
        FakeMessage(
            content=[FakeToolUseBlock(id="tu_1", name="echo", input={"foo": "bar"})],
            stop_reason="tool_use",
        ),
        _end_turn("Done."),
    ])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["do it"], descriptions=DESCRIPTIONS, run_id="run-2", client=client,
    )

    assert len(calls) == 1
    registry, tool_name, principal, run_id, kwargs = calls[0]
    assert registry is FAKE_REGISTRY
    assert tool_name == "echo"
    assert principal is CUSTOMER
    assert run_id == "run-2"
    assert kwargs == {"foo": "bar"}

    tool_call = trace.turns[1].tool_calls[0]
    assert tool_call.tool == "echo"
    assert tool_call.decision == "executed"
    assert tool_call.entity_ref == "customer:1"
    assert tool_call.declared_tier == 0


def test_find_my_account_resolution_updates_principal_for_the_rest_of_the_run():
    """find_my_account_tool's own docstring: 'customer_id in the return value is for the
    harness to update its own session-level principal with' -- this is that update. Without it,
    a later tool call in the same conversation would still run as the original unresolved
    principal even after identity successfully resolved."""
    seen_principal_ids = []
    registry = {
        "find_my_account": ToolSpec(
            fn=lambda **kw: {"decision": "executed", "resolved": True, "customer_id": 99}, tier=0,
        ),
        "whoami": ToolSpec(
            fn=lambda *, principal, run_id=None, **kw: (
                seen_principal_ids.append(principal.id) or {"decision": "executed"}
            ),
            tier=0,
        ),
    }
    descriptions = {"find_my_account": "Resolve identity.", "whoami": "Echo back the principal."}
    client = FakeAnthropicClient([
        FakeMessage(
            content=[FakeToolUseBlock(id="tu_1", name="find_my_account", input={
                "name": "A", "email": "a@example.com", "phone": "1", "address": "x",
            })],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeToolUseBlock(id="tu_2", name="whoami", input={})], stop_reason="tool_use"),
        _end_turn("Done."),
    ])
    run_agent(
        registry=registry, principal=Principal(type="customer", id=None), system_prompt="sys",
        user_turns=["hi"], descriptions=descriptions, run_id="run-resolve", client=client,
    )
    assert seen_principal_ids == [99]


def test_find_my_account_unresolved_leaves_principal_unchanged():
    seen_principal_ids = []
    registry = {
        "find_my_account": ToolSpec(
            fn=lambda **kw: {"decision": "executed", "resolved": False}, tier=0,
        ),
        "whoami": ToolSpec(
            fn=lambda *, principal, run_id=None, **kw: (
                seen_principal_ids.append(principal.id) or {"decision": "executed"}
            ),
            tier=0,
        ),
    }
    descriptions = {"find_my_account": "Resolve identity.", "whoami": "Echo back the principal."}
    client = FakeAnthropicClient([
        FakeMessage(
            content=[FakeToolUseBlock(id="tu_1", name="find_my_account", input={
                "name": "A", "email": "a@example.com", "phone": "1", "address": "x",
            })],
            stop_reason="tool_use",
        ),
        FakeMessage(content=[FakeToolUseBlock(id="tu_2", name="whoami", input={})], stop_reason="tool_use"),
        _end_turn("Done."),
    ])
    run_agent(
        registry=registry, principal=Principal(type="customer", id=None), system_prompt="sys",
        user_turns=["hi"], descriptions=descriptions, run_id="run-unresolved", client=client,
    )
    assert seen_principal_ids == [None]


def test_parallel_tool_calls_are_batched_into_a_single_user_message(monkeypatch):
    monkeypatch.setattr(
        "agent.loop.dispatch",
        lambda registry, tool_name, principal, *, run_id=None, **kwargs: {"decision": "executed"},
    )
    client = FakeAnthropicClient([
        FakeMessage(
            content=[
                FakeToolUseBlock(id="tu_1", name="echo", input={"a": 1}),
                FakeToolUseBlock(id="tu_2", name="echo", input={"a": 2}),
            ],
            stop_reason="tool_use",
        ),
        _end_turn("Both done."),
    ])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["do both"], descriptions=DESCRIPTIONS, run_id="run-3", client=client,
    )

    # second API call's last message is the tool-results turn -- must carry both results
    second_call_messages = client.calls[1]["messages"]
    tool_result_message = second_call_messages[-1]
    assert tool_result_message["role"] == "user"
    assert len(tool_result_message["content"]) == 2
    assert {r["tool_use_id"] for r in tool_result_message["content"]} == {"tu_1", "tu_2"}


def test_a_raising_tool_produces_an_is_error_result_instead_of_crashing(monkeypatch):
    def exploding_dispatch(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("agent.loop.dispatch", exploding_dispatch)
    client = FakeAnthropicClient([
        FakeMessage(content=[FakeToolUseBlock(id="tu_1", name="echo", input={})], stop_reason="tool_use"),
        _end_turn("Sorted it out."),
    ])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["go"], descriptions=DESCRIPTIONS, run_id="run-4", client=client,
    )

    assert trace.outcome == "ok"  # a tool error is not a harness error
    tool_result_message = client.calls[1]["messages"][-1]
    assert tool_result_message["content"][0]["is_error"] is True
    assert "boom" in tool_result_message["content"][0]["content"]


def test_max_turns_cap_stops_the_loop_and_records_it(monkeypatch):
    monkeypatch.setattr(
        "agent.loop.dispatch",
        lambda registry, tool_name, principal, *, run_id=None, **kwargs: {"decision": "executed"},
    )
    # An endless supply of tool_use turns -- the model that never stops calling tools.
    responses = [
        FakeMessage(content=[FakeToolUseBlock(id=f"tu_{i}", name="echo", input={})], stop_reason="tool_use")
        for i in range(50)
    ]
    client = FakeAnthropicClient(responses)
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["go"], descriptions=DESCRIPTIONS, run_id="run-5", client=client, max_turns=3,
    )
    assert trace.hit_turn_cap is True
    assert len(client.calls) == 3  # never exceeds max_turns


def test_scripted_user_turns_are_played_in_order_and_never_replayed():
    client = FakeAnthropicClient([_end_turn("First answer."), _end_turn("Second answer.")])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["First question.", "Second question."], descriptions=DESCRIPTIONS,
        run_id="run-6", client=client,
    )
    roles_and_text = [(t.role, t.text) for t in trace.turns]
    assert roles_and_text == [
        ("user", "First question."),
        ("assistant", "First answer."),
        ("user", "Second question."),
        ("assistant", "Second answer."),
    ]
    assert len(client.calls) == 2  # no third call once scripted turns are exhausted


@pytest.mark.parametrize(
    "make_exception",
    [
        lambda req: anthropic.NotFoundError("not found", response=httpx2.Response(404, request=req), body=None),
        lambda req: anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=req), body=None),
        lambda req: anthropic.APIStatusError("server error", response=httpx2.Response(500, request=req), body=None),
        lambda req: anthropic.APIConnectionError(request=req),
    ],
)
def test_sdk_exceptions_degrade_to_a_harness_error_outcome_not_a_crash(make_exception):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    exc = make_exception(request)

    class RaisingClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise exc

    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["hello"], descriptions=DESCRIPTIONS, run_id="run-7", client=RaisingClient(),
    )
    assert trace.outcome == "harness_error"
    assert trace.error_detail is not None
    assert trace.error_detail.startswith(type(exc).__name__ + ":")


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_truncated_or_refused_turns_are_recorded_not_silently_treated_as_end_turn(stop_reason):
    """A max_tokens or refusal stop_reason falls into the same code path as a normal end_turn
    (this loop still ends the conversation), but must be visible on the trace rather than
    indistinguishable from a clean completion."""
    client = FakeAnthropicClient([FakeMessage(content=[FakeTextBlock(text="...")], stop_reason=stop_reason)])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["hello"], descriptions=DESCRIPTIONS, run_id="run-8", client=client,
    )
    assert trace.outcome == "ok"
    assert trace.turns[-1].stop_reason == stop_reason


def test_normal_end_turn_leaves_stop_reason_unset():
    client = FakeAnthropicClient([_end_turn("All good.")])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["hello"], descriptions=DESCRIPTIONS, run_id="run-9", client=client,
    )
    assert trace.turns[-1].stop_reason is None


def test_forbidden_arg_keys_are_stripped_before_reaching_dispatch(monkeypatch):
    captured_kwargs = {}

    def spy_dispatch(registry, tool_name, principal, *, run_id=None, **kwargs):
        captured_kwargs.update(kwargs)
        return {"decision": "executed"}

    monkeypatch.setattr("agent.loop.dispatch", spy_dispatch)
    client = FakeAnthropicClient([
        FakeMessage(
            content=[FakeToolUseBlock(
                id="tu_1", name="echo",
                input={"foo": "bar", "principal": "sneaky", "run_id": "sneaky-run"},
            )],
            stop_reason="tool_use",
        ),
        _end_turn("Done."),
    ])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["go"], descriptions=DESCRIPTIONS, run_id="run-8", client=client,
    )
    assert captured_kwargs == {"foo": "bar"}


def test_usage_accumulates_across_multiple_api_calls(monkeypatch):
    monkeypatch.setattr(
        "agent.loop.dispatch",
        lambda registry, tool_name, principal, *, run_id=None, **kwargs: {"decision": "executed"},
    )
    client = FakeAnthropicClient([
        FakeMessage(
            content=[FakeToolUseBlock(id="tu_1", name="echo", input={})], stop_reason="tool_use",
            usage=FakeUsage(input_tokens=100, output_tokens=20, cache_read_input_tokens=50),
        ),
        FakeMessage(
            content=[FakeTextBlock(text="done")], stop_reason="end_turn",
            usage=FakeUsage(input_tokens=10, output_tokens=5, cache_read_input_tokens=200),
        ),
    ])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["go"], descriptions=DESCRIPTIONS, run_id="run-9", client=client, model="claude-sonnet-5",
    )
    assert trace.usage.input_tokens == 110
    assert trace.usage.output_tokens == 25
    assert trace.usage.cache_read_input_tokens == 250
    assert trace.usage.cost_usd > 0


def test_system_prompt_carries_a_cache_control_breakpoint():
    """A cache breakpoint anywhere but the end of the (frozen) system prompt silently loses the
    cache across a batch -- assert it's actually there on every request the loop sends."""
    client = FakeAnthropicClient([_end_turn("ok")])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["hi"], descriptions=DESCRIPTIONS, run_id="run-10", client=client,
    )
    system = client.calls[0]["system"]
    assert system[-1]["cache_control"] == {"type": "ephemeral"}
    assert system[-1]["text"] == "You are helpful."


def test_context_note_appended_as_a_second_uncached_system_block():
    """context_note must never be merged into the frozen system_prompt block (that would
    invalidate its cache_control breakpoint across every other case in a batch) and must never
    be prepended as a fake user turn (see run_agent's own no-simulated-user docstring) -- it's a
    second block, appended after, with no cache_control of its own."""
    client = FakeAnthropicClient([_end_turn("ok")])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["hi"], descriptions=DESCRIPTIONS, run_id="run-note",
        context_note="Today is Tuesday.", client=client,
    )
    system = client.calls[0]["system"]
    assert system[0] == {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}
    assert system[1] == {"type": "text", "text": "Today is Tuesday."}
    assert len(system) == 2


def test_no_context_note_leaves_system_byte_identical_to_before():
    """Every existing caller passes no context_note -- confirm the default produces exactly the
    single-block system list run_agent sent before context_note existed, not a list with an
    empty second block."""
    client = FakeAnthropicClient([_end_turn("ok")])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["hi"], descriptions=DESCRIPTIONS, run_id="run-no-note", client=client,
    )
    assert client.calls[0]["system"] == [
        {"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}},
    ]


def test_tools_sent_on_every_request_match_build_schemas_output():
    from agent.schemas import build_schemas

    client = FakeAnthropicClient([_end_turn("ok")])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["hi"], descriptions=DESCRIPTIONS, run_id="run-11", client=client,
    )
    assert client.calls[0]["tools"] == build_schemas(FAKE_REGISTRY, DESCRIPTIONS)


def test_no_temperature_or_top_p_sent_on_any_request():
    """temperature/top_p are removed on Sonnet 5 and Opus 5 -- sending either is a 400. Guard
    against ever reintroducing one."""
    client = FakeAnthropicClient([_end_turn("ok")])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["hi"], descriptions=DESCRIPTIONS, run_id="run-12", client=client,
    )
    assert "temperature" not in client.calls[0]
    assert "top_p" not in client.calls[0]
    assert "budget_tokens" not in str(client.calls[0].get("thinking", {}))


# --- per-model quality knobs -----------------------------------------------------------------
#
# Haiku 4.5 rejects BOTH of the Claude 5 knobs, independently and with a 400 each. Verified
# against the live API: adaptive thinking answers "adaptive thinking is not supported on this
# model", and with thinking removed, effort answers "This model does not support the effort
# parameter". The whole small-model ablation arm harness-errored on all 70 cases because of it.


def _one_call(model):
    client = FakeAnthropicClient([_end_turn("ok")])
    run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["hi"], descriptions=DESCRIPTIONS, run_id="run-1", client=client,
        model=model,
    )
    return client.calls[0]


def test_claude_5_still_gets_adaptive_thinking_and_effort():
    """The ablation must not quietly change the arm it is compared against: this path has to stay
    exactly what it was before the knobs became model-dependent."""
    call = _one_call("claude-sonnet-5")

    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}


def test_a_model_without_adaptive_thinking_gets_neither_rejected_parameter():
    call = _one_call("claude-haiku-4-5-20251001")

    assert "output_config" not in call
    assert call["thinking"]["type"] == "enabled"
    assert call["thinking"]["budget_tokens"] < call["max_tokens"]


def test_the_trace_records_the_effort_actually_sent_not_the_one_requested():
    """A Haiku trace claiming effort=high would describe a parameter the API refused, and would
    make the run look config-compatible with a Sonnet run under aggregate_runs' triple."""
    client = FakeAnthropicClient([_end_turn("ok")])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="sys",
        user_turns=["hi"], descriptions=DESCRIPTIONS, run_id="run-1", client=client,
        model="claude-haiku-4-5-20251001", effort="high",
    )

    assert trace.effort == "budget_tokens:4000"


# --- the confirmation affordance -----------------------------------------------------------
#
# An agent that correctly ends its turn asking "shall I go ahead and book this?" gets no answer
# from a fixed script, the run ends, and the case fails require_decision for behaving well. A
# case may supply `on_confirmation_request` to answer that once. These tests pin the three
# properties that keep it from becoming a way to launder a passing grade: it never pre-empts a
# scripted line, it never fires unprompted, and it never fires twice.


def _asks(text: str) -> FakeMessage:
    return FakeMessage(content=[FakeTextBlock(text=text)], stop_reason="end_turn")


def test_confirmation_affordance_answers_a_trailing_question():
    client = FakeAnthropicClient([
        _asks("I have everything I need. Shall I go ahead and book it?"),
        _end_turn("Booked."),
    ])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["Book me a slot."], descriptions=DESCRIPTIONS, run_id="run-aff-1",
        client=client, on_confirmation_request="Yes, go ahead.",
    )
    assert [t.role for t in trace.turns] == ["user", "assistant", "user", "assistant"]
    assert trace.turns[2].text == "Yes, go ahead."
    assert trace.turns[2].source == "affordance"
    assert trace.turns[0].source == "scripted"


def test_confirmation_affordance_recognizes_a_request_that_is_not_a_question():
    """prov_01's agent closed with "Please confirm and I'll finalize the booking!" -- a request
    for a go-ahead that happens not to be grammatically a question."""
    client = FakeAnthropicClient([
        _asks("Here are the details. Please confirm and I'll finalize the booking!"),
        _end_turn("Done."),
    ])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["Book me a slot."], descriptions=DESCRIPTIONS, run_id="run-aff-2",
        client=client, on_confirmation_request="Yes, that is all correct.",
    )
    assert [t.role for t in trace.turns] == ["user", "assistant", "user", "assistant"]
    assert trace.turns[2].source == "affordance"


def test_confirmation_affordance_does_not_fire_when_the_agent_asked_nothing():
    """The agent finished its work and closed. Nothing is waiting on the caller, so injecting a
    reply would be inventing conversation, not repairing a stall."""
    client = FakeAnthropicClient([_end_turn("All set -- you're booked for Tuesday at 10am.")])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["Book me a slot."], descriptions=DESCRIPTIONS, run_id="run-aff-3",
        client=client, on_confirmation_request="Yes, go ahead.",
    )
    assert [t.role for t in trace.turns] == ["user", "assistant"]
    assert len(client.calls) == 1


def test_scripted_turns_always_take_precedence_over_the_affordance():
    """The affordance is a fallback for an exhausted script, never a substitute for the next
    scripted line -- otherwise a case's own turns would stop being what it tests."""
    client = FakeAnthropicClient([
        _asks("Shall I go ahead?"),
        _asks("And shall I confirm that too?"),
        _end_turn("Done."),
    ])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["Book me a slot.", "Actually, make it Thursday."],
        descriptions=DESCRIPTIONS, run_id="run-aff-4",
        client=client, on_confirmation_request="Yes, go ahead.",
    )
    user_turns = [(t.text, t.source) for t in trace.turns if t.role == "user"]
    assert user_turns == [
        ("Book me a slot.", "scripted"),
        ("Actually, make it Thursday.", "scripted"),
        ("Yes, go ahead.", "affordance"),
    ]


def test_confirmation_affordance_fires_at_most_once_per_run():
    """A model that keeps asking must not keep being answered -- that is a runaway loop with
    the harness holding one end of it."""
    client = FakeAnthropicClient([
        _asks("Shall I go ahead?"),
        _asks("Sorry, just to confirm once more -- shall I?"),
    ])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["Book me a slot."], descriptions=DESCRIPTIONS, run_id="run-aff-5",
        client=client, on_confirmation_request="Yes, go ahead.",
    )
    assert sum(1 for t in trace.turns if t.source == "affordance") == 1
    assert len(client.calls) == 2


def test_a_case_without_the_affordance_is_unaffected():
    """The default path has to stay exactly what it was: no reply, conversation ends."""
    client = FakeAnthropicClient([_asks("Shall I go ahead and book it?")])
    trace = run_agent(
        registry=FAKE_REGISTRY, principal=CUSTOMER, system_prompt="You are helpful.",
        user_turns=["Book me a slot."], descriptions=DESCRIPTIONS, run_id="run-aff-6",
        client=client,
    )
    assert [t.role for t in trace.turns] == ["user", "assistant"]
    assert all(t.source == "scripted" for t in trace.turns)
    assert len(client.calls) == 1
