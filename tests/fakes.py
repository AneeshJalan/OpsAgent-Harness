"""Lightweight stand-ins for the Anthropic SDK's response shapes, used only by test_loop.py.
Mirrors just the attributes agent/loop.py actually reads -- block.type/.text/.name/.input/.id,
response.content/.stop_reason/.usage, usage.input_tokens/.output_tokens/.cache_*_input_tokens --
never the real SDK, so no network access and no API key are needed to test the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeThinkingBlock:
    thinking: str
    signature: str = "sig"
    type: str = "thinking"


@dataclass
class FakeRedactedThinkingBlock:
    data: str = "encrypted"
    type: str = "redacted_thinking"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass
class FakeMessage:
    content: list[Any]
    stop_reason: str
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeAnthropicClient:
    """Plays back a scripted list of FakeMessage responses, one per call to
    `client.messages.create(...)`, and records every call's kwargs so a test can assert on what
    the loop actually sent (system/tools/messages)."""

    def __init__(self, responses: list[FakeMessage]):
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.messages = self  # client.messages.create(...) resolves to self.create(...)

    def create(self, **kwargs: Any) -> FakeMessage:
        # Snapshot `messages` at call time -- the caller keeps mutating that same list object
        # (appending later turns) after this call returns, so recording the live reference
        # would make every earlier call's recorded history silently drift to the final state.
        snapshot = {**kwargs, "messages": list(kwargs.get("messages", []))}
        self.calls.append(snapshot)
        if not self._responses:
            raise AssertionError("FakeAnthropicClient ran out of scripted responses")
        return self._responses.pop(0)


class AutoEndTurnClient:
    """A fake client that never runs out: every call returns a plain end_turn text response,
    regardless of how many scripted user turns a case has or how many cases share one instance.
    Used for orchestration-level tests (e.g. run_suite.py looping over many cases) where the
    point is proving the loop/aggregation wiring works, not re-proving any individual case's
    guard/scored correctness against a precisely scripted conversation -- that's what
    test_case_runner.py's precisely-scripted FakeAnthropicClient tests already cover."""

    def __init__(self, text: str = "Sure, happy to help."):
        self.calls: list[dict[str, Any]] = []
        self.messages = self
        self._text = text

    def create(self, **kwargs: Any) -> FakeMessage:
        self.calls.append({**kwargs, "messages": list(kwargs.get("messages", []))})
        return FakeMessage(content=[FakeTextBlock(text=self._text)], stop_reason="end_turn")
