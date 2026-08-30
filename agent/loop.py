"""The manual tool-use agent loop. One function, parameterized by registry + system prompt +
principal -- never two separate loops for the two personas.

Written by hand against `client.messages.create` rather than the SDK's beta tool runner: the
tool runner wants tools as decorated Python functions it calls directly, and this project's
entire thesis is that every call goes through `dispatch()`, which needs an out-of-band
`principal` the model must never see. Wrapping 28 registry entries in per-run closures just to
smuggle the principal past a helper -- and taking a beta dependency to do it -- buys nothing:
this loop is about a hundred lines, and every call has to be intercepted anyway to write the
trace.

`principal` isn't fixed for the whole run: a successful find_my_account call resolves it, and
every dispatch() call for the rest of the conversation runs as that resolved customer from then
on -- see the update right after dispatch() in the tool-call loop below. Without this, identity
resolution would be theater: the caller could prove who they are and every subsequent call
would still run as the original unresolved principal.

Model note (current as of this file's writing): `temperature` is removed on Claude Opus 5,
Sonnet 5, and the 4.6+ family -- there is no `temperature=0` determinism lever, so
`output_config: {effort: ...}` plus `thinking: {type: "adaptive"}` is the quality knob instead.
Default model is `claude-sonnet-5` ($3/$15 per MTok, $2/$10 intro through 2026-08-31) --
deployment-realistic, and it leaves `claude-opus-5` free to be a strictly stronger judge on a
different model family than whatever's under test.
"""

from __future__ import annotations

import json
import re
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


# A turn that hands the conversation back to the caller. The question mark carries almost all
# of this on its own; the cue list covers the handful of endings that ask for a go-ahead without
# grammatically being a question ("Please confirm and I'll finalize the booking!"). Deliberately
# anchored near the end of the turn -- an assistant that asks something mid-paragraph and then
# closes with a statement is not waiting on anyone.
_REQUEST_CUE_RE = re.compile(
    r"\b(please confirm|confirm and i'?ll|let me know|just say the word|say the word|"
    r"ready when you are|if you'?d like me to)\b",
    re.IGNORECASE,
)
_REQUEST_CUE_WINDOW = 300


def _invites_a_reply(text: str) -> bool:
    tail = text.strip()
    if not tail:
        return False
    if tail.endswith("?"):
        return True
    return bool(_REQUEST_CUE_RE.search(tail[-_REQUEST_CUE_WINDOW:]))


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
    context_note: str | None = None,
    on_confirmation_request: str | None = None,
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

    `on_confirmation_request` is the one bounded exception, and it exists because that rule has
    a failure mode: an agent that correctly asks "shall I go ahead and book this?" as its last
    act gets no answer, the run ends, and the case fails `require_decision` for doing the right
    thing. One full suite run had six cases failing that way -- every one of them ending on an
    unanswered confirmation, with the booking never attempted. When a case supplies this string,
    it is played **at most once**, and only when all three hold: the scripted lines are
    exhausted, the assistant's closing turn actually invites a reply (`_invites_a_reply`), and
    the affordance has not already been spent. It is opt-in per case, so a case whose premise is
    that the caller goes silent simply omits it. This is not a simulated user -- it introduces no
    second model and no generated text; the reply is a fixed string the case author wrote,
    played conditionally rather than unconditionally. That conditionality is the entire point:
    appending it to `turns` instead would fire it even when the agent had already acted
    correctly, which for a booking case means a second, spurious booking.

    This is eval-replay only, not a live interactive agent: there is no other agent-loop entry
    point in this codebase, but reusing this function to serve a real conversation would silently
    inherit "ignore what the assistant said, just play the next scripted line" -- a real
    interactive agent needs its own entry point (a live per-turn input callback in place of
    `user_turns`), not a repurposing of this one.

    Every tool call is executed through `dispatch(registry, name, principal, run_id=run_id,
    **args)` -- never a registry function directly, so every call is audited and role-gated
    exactly as it would be for any other caller. All `tool_result` blocks produced by one
    assistant turn are returned in a single following user message; Opus 5 (and Sonnet 5) can
    emit parallel `tool_use` blocks, and splitting their results across multiple messages
    silently trains the model to stop calling tools in parallel.

    `context_note`, when given, is appended as a second, uncached `system` content block after
    the frozen `system_prompt` block -- never merged into `system_prompt` itself, and never
    prepended as a fake user turn. Per-run facts (today's date, whether this caller is already
    verified) belong here, not in `system_prompt` (see agent/prompts.py's own docstring on why
    that string must stay a frozen constant) and not in `user_turns` (which stays exactly what
    the case script says, per the no-simulated-user principle above). Keeping `system_prompt`'s
    bytes unchanged means its `cache_control: ephemeral` breakpoint still hits across every run
    in a batch; only this small second block is new per run.
    """
    client = client or anthropic.Anthropic()
    tools = build_schemas(registry, descriptions)
    system_blocks: list[dict[str, Any]] = [
        {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
    ]
    if context_note:
        system_blocks.append({"type": "text", "text": context_note})

    trace = Trace(
        run_id=run_id, case_id=case_id, persona=persona, variant=variant,
        model=model, effort=effort, replicate=replicate,
        principal={"type": principal.type, "id": principal.id},
    )

    messages: list[dict[str, Any]] = []
    remaining_user_turns = list(user_turns)
    confirmation_affordance = on_confirmation_request  # consumed at most once, see below
    usage_totals = {
        "input_tokens": 0, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0, "output_tokens": 0,
    }

    def _play_next_user_turn(last_assistant_text: str | None = None) -> bool:
        """Scripted lines first, always, in order. Only once they run out is the confirmation
        affordance considered -- so a case that supplies one is scored on exactly the script it
        would have been scored on otherwise, right up until the point the conversation would
        have died."""
        nonlocal confirmation_affordance
        if remaining_user_turns:
            next_text = remaining_user_turns.pop(0)
            messages.append({"role": "user", "content": next_text})
            trace.turns.append(TurnRecord(role="user", text=next_text))
            return True

        if (
            confirmation_affordance is not None
            and last_assistant_text is not None
            and _invites_a_reply(last_assistant_text)
        ):
            reply, confirmation_affordance = confirmation_affordance, None  # once per run
            messages.append({"role": "user", "content": reply})
            trace.turns.append(TurnRecord(role="user", text=reply, source="affordance"))
            return True

        return False

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
                system=system_blocks,
                tools=tools,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                messages=messages,
            )
            _accumulate_usage(usage_totals, response.usage)

            assistant_turn = TurnRecord(role="assistant", text=_extract_text(response.content))
            if response.stop_reason in ("max_tokens", "refusal"):
                # Recorded so a truncated or refused turn is visible in the trace instead of
                # looking identical to a normal completion -- worth knowing about even though
                # this loop still treats it as an end-of-turn below, same as end_turn.
                assistant_turn.stop_reason = response.stop_reason
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = _tool_use_blocks(response.content)
            if response.stop_reason == "tool_use":
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

                    # find_my_account_tool's own docstring: "customer_id in the return value is
                    # for the harness to update its own session-level principal with (out-of-
                    # band, never re-submitted by the model as an argument to anything)". This is
                    # that update -- every dispatch() call for the rest of this run now runs as
                    # the resolved customer, exactly as if the harness had known who this was
                    # from the start. Without it, a later tool call in the same conversation
                    # (e.g. get_my_appointments) would still see the original unresolved
                    # principal and deny, even though identity was just successfully resolved.
                    if not is_error and block.name == "find_my_account" and result.get("resolved"):
                        principal = Principal(type="customer", id=result["customer_id"])

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

            if _play_next_user_turn(assistant_turn.text):
                continue

            break  # end_turn (or max_tokens/refusal) -- nothing left to say, done

    except anthropic.NotFoundError as exc:
        trace.outcome = "harness_error"
        trace.error_detail = f"{type(exc).__name__}: {exc}"
    except anthropic.RateLimitError as exc:
        trace.outcome = "harness_error"
        trace.error_detail = f"{type(exc).__name__}: {exc}"
    except anthropic.APIStatusError as exc:
        trace.outcome = "harness_error"
        trace.error_detail = f"{type(exc).__name__}: {exc}"
    except anthropic.APIConnectionError as exc:
        trace.outcome = "harness_error"
        trace.error_detail = f"{type(exc).__name__}: {exc}"

    trace.wall_ms = (time.monotonic() - start_wall) * 1000
    usage = UsageRecord(**usage_totals)
    usage.cost_usd = compute_cost_usd(model, usage)
    trace.usage = usage
    return trace
