"""The manual tool-use agent loop. One function, parameterized by registry + system prompt +
principal -- never two separate loops for the two personas.

Written by hand against `client.messages.create` rather than the SDK's beta tool runner: the
tool runner wants tools as decorated Python functions it calls directly, and this project's
entire thesis is that every call goes through `dispatch()`, which needs an out-of-band
`principal` the model must never see. Wrapping 28 registry entries in per-run closures just to
smuggle the principal past a helper -- and taking a beta dependency to do it -- buys nothing:
this loop is about a hundred lines, and every call has to be intercepted anyway to write the
trace.

Model note (current as of this file's writing): `temperature` is removed on Claude Opus 5,
Sonnet 5, and the 4.6+ family -- there is no `temperature=0` determinism lever, so
`output_config: {effort: ...}` plus `thinking: {type: "adaptive"}` is the quality knob instead.
Default model is `claude-sonnet-5` ($3/$15 per MTok, $2/$10 intro through 2026-08-31) --
deployment-realistic, and it leaves `claude-opus-5` free to be a strictly stronger judge on a
different model family, per Day 3.
"""

from __future__ import annotations

import json
import time
from typing import Any

import anthropic
from dotenv import load_dotenv

from agent.schemas import build_schemas
from agent.trace import ToolCallRecord, Trace, TurnRecord, UsageRecord, compute_cost_usd
from tools.dispatcher import Registry, dispatch
from tools.principal import Principal

# Populates os.environ from a local .env file, if one exists (see .example.env for the expected
# format) -- a no-op when it doesn't, so this is safe to import in CI or any environment where
# ANTHROPIC_API_KEY is already set another way. Anthropic() itself reads the key straight out of
# the environment; this just gets it there for local development.
load_dotenv()

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 16000

# Belt-and-suspenders: strict tool schemas (agent/schemas.py) already make it impossible for the
# model to submit these as arguments, but a tool call's args dict is stripped of them again here
# before ever reaching dispatch() -- the out-of-band principal/run_id boundary is worth
# defending at more than one layer, the same way ownership scoping is checked once by the
# dispatcher's registry gate and again inside each tool function.
_FORBIDDEN_ARG_KEYS = ("principal", "run_id")


def _extract_text(content: list[Any]) -> str:
    return "".join(block.text for block in content if block.type == "text")


def _tool_use_blocks(content: list[Any]) -> list[Any]:
    return [block for block in content if block.type == "tool_use"]


def _parse_tool_input(raw: Any) -> dict[str, Any]:
    """Never string-match a tool call's serialized input -- Opus 5 (and the rest of the 4.6+
    family) varies its JSON escaping. json.loads is the only correct way to read it. The SDK
    usually hands back an already-parsed dict; only fall through to json.loads for a raw string,
    which covers a fake/mock client in tests as well as any future SDK change."""
    if isinstance(raw, dict):
        return dict(raw)
    return json.loads(raw)


def _accumulate_usage(totals: dict[str, int], usage: Any) -> None:
    totals["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    totals["cache_read_input_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    totals["cache_creation_input_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
    totals["output_tokens"] += getattr(usage, "output_tokens", 0) or 0


def run_agent(
    *,
    registry: Registry,
    principal: Principal,
    system_prompt: str,
    user_turns: list[str],
    descriptions: dict[str, str],
    run_id: str,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
    max_turns: int = 12,
    effort: str = "high",
    case_id: str = "",
    persona: str = "",
    variant: str = "baseline",
    replicate: int = 0,
) -> Trace:
    """Runs one scripted conversation end to end and returns its Trace.

    `user_turns` is a fixed ordered list -- turn k+1 is played once the assistant's
    current turn ends (no more tool calls), regardless of what the assistant said. No simulated
    user: the model is already the only source of non-determinism in this harness, and doubling
    it would make a failing case impossible to attribute.

    Every tool call is executed through `dispatch(registry, name, principal, run_id=run_id,
    **args)` -- never a registry function directly, so every call is audited and role-gated
    exactly as it would be for any other caller. All `tool_result` blocks produced by one
    assistant turn are returned in a single following user message; Opus 5 (and Sonnet 5) can
    emit parallel `tool_use` blocks, and splitting their results across multiple messages
    silently trains the model to stop calling tools in parallel.
    """
    client = client or anthropic.Anthropic()
    tools = build_schemas(registry, descriptions)

    trace = Trace(
        run_id=run_id, case_id=case_id, persona=persona, variant=variant,
        model=model, effort=effort, replicate=replicate,
        principal={"type": principal.type, "id": principal.id},
    )

    messages: list[dict[str, Any]] = []
    remaining_user_turns = list(user_turns)
    usage_totals = {
        "input_tokens": 0, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0, "output_tokens": 0,
    }

    def _play_next_user_turn() -> bool:
        if not remaining_user_turns:
            return False
        next_text = remaining_user_turns.pop(0)
        messages.append({"role": "user", "content": next_text})
        trace.turns.append(TurnRecord(role="user", text=next_text))
        return True

    _play_next_user_turn()  # seed the conversation -- there is always at least one user turn

    start_wall = time.monotonic()
    turns_used = 0
    try:
        while True:
            turns_used += 1
            if turns_used > max_turns:
                trace.hit_turn_cap = True
                break

            response = client.messages.create(
                model=model,
                max_tokens=DEFAULT_MAX_TOKENS,
                system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
                tools=tools,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                messages=messages,
            )
            _accumulate_usage(usage_totals, response.usage)

            assistant_turn = TurnRecord(role="assistant", text=_extract_text(response.content))
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = _tool_use_blocks(response.content)
            if tool_use_blocks:
                tool_results = []
                for block in tool_use_blocks:
                    args = _parse_tool_input(block.input)
                    for forbidden in _FORBIDDEN_ARG_KEYS:
                        args.pop(forbidden, None)

                    tool_start = time.monotonic()
                    try:
                        result = dispatch(registry, block.name, principal, run_id=run_id, **args)
                        is_error = False
                    except Exception as exc:  # a tool that raises still gets a tool_result
                        result = {"error": str(exc)}
                        is_error = True
                    latency_ms = (time.monotonic() - tool_start) * 1000

                    declared_tier = registry[block.name].tier if block.name in registry else -1
                    assistant_turn.tool_calls.append(ToolCallRecord(
                        tool=block.name, args=args, declared_tier=declared_tier, result=result,
                        decision=result.get("decision"), reason=result.get("reason"),
                        entity_ref=result.get("entity_ref"), latency_ms=latency_ms,
                    ))
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                        "is_error": is_error,
                    })

                trace.turns.append(assistant_turn)
                messages.append({"role": "user", "content": tool_results})  # one message, every result
                continue  # more of the model's turn to play out before any scripted user line

            trace.turns.append(assistant_turn)

            if response.stop_reason == "pause_turn":
                # A server-side tool paused mid-turn. Nothing in either registry uses one today,
                # but resuming (rather than treating it as final) matches the SDK's documented
                # pattern and costs nothing when it never triggers.
                continue

            if _play_next_user_turn():
                continue

            break  # end_turn, no more scripted turns -- conversation is done

    except anthropic.NotFoundError:
        trace.outcome = "harness_error"
    except anthropic.RateLimitError:
        trace.outcome = "harness_error"
    except anthropic.APIStatusError:
        trace.outcome = "harness_error"
    except anthropic.APIConnectionError:
        trace.outcome = "harness_error"

    trace.wall_ms = (time.monotonic() - start_wall) * 1000
    usage = UsageRecord(**usage_totals)
    usage.cost_usd = compute_cost_usd(model, usage)
    trace.usage = usage
    return trace
