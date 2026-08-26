"""The structured trace: one JSON file per run, at evals/runs/<run_id>/trace.json. Every
checker in evals/checks/ reads this file (plus the state_before.json/state_after.json snapshots
that sit beside it) instead of re-deriving anything from a live conversation, so the trace shape
defined here is the actual contract between the agent loop and every checker downstream.

`decision` and `reason` on a ToolCallRecord are copied straight from the tool's returned dict --
already plain strings by the time they get here (tools/dispatcher.py and every registry tool
normalize Decision/Reason enum members to `.value` before returning), never re-wrapped.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_RUNS_DIR = Path("evals/runs")

# Published per-MTok rates this project bills runs against. Sonnet 5 carries an intro rate
# through 2026-08-31 (see agent/loop.py's module docstring) -- update the tuple below, not the
# lookup logic, once that expires. Cache reads bill at ~0.1x the input rate and cache writes at
# ~1.25x; costing from input_tokens alone makes every cached run look free, which is exactly the
# blind spot this table exists to close.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),  # intro rate through 2026-08-31; standard rate is 3.00/15.00
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
_CACHE_READ_MULTIPLIER = 0.1
_CACHE_WRITE_MULTIPLIER = 1.25


@dataclass
class ToolCallRecord:
    tool: str
    args: dict[str, Any]
    declared_tier: int
    result: dict[str, Any]
    decision: str | None
    reason: str | None
    entity_ref: str | None
    latency_ms: float


@dataclass
class TurnRecord:
    role: str  # "assistant" | "user"
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    # Only ever set on an assistant turn, and only when the API's stop_reason was something
    # other than "tool_use"/"end_turn"/"pause_turn" -- a truncated ("max_tokens") or refused
    # ("refusal") turn would otherwise look identical to a normal completion in the trace.
    stop_reason: str | None = None


@dataclass
class UsageRecord:
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class Trace:
    run_id: str
    case_id: str
    persona: str
    variant: str
    model: str
    effort: str
    replicate: int
    principal: dict[str, Any]
    turns: list[TurnRecord] = field(default_factory=list)
    usage: UsageRecord = field(default_factory=UsageRecord)
    hit_turn_cap: bool = False
    outcome: str = "ok"  # "ok" | "harness_error" -- a harness_error is excluded from pass rates
    # Populated only when outcome == "harness_error" -- f"{type(exc).__name__}: {exc}" for
    # whichever typed SDK exception ended the run, so a harness_error is diagnosable from the
    # trace alone instead of needing to reproduce it.
    error_detail: str | None = None
    wall_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def run_dir(self, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> Path:
        return Path(runs_dir) / self.run_id

    def write(self, runs_dir: Path | str = DEFAULT_RUNS_DIR) -> Path:
        out_dir = self.run_dir(runs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "trace.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        return path


def compute_cost_usd(model: str, usage: UsageRecord) -> float:
    """Cost from all four usage counters, not just input_tokens. An unrecognized model returns
    0.0 rather than guessing a rate -- a silently wrong cost figure is worse than a visibly
    missing one, and the caller can tell the two apart."""
    rates = PRICING_PER_MTOK.get(model)
    if rates is None:
        return 0.0
    input_rate, output_rate = rates
    cost = (
        usage.input_tokens * input_rate
        + usage.cache_read_input_tokens * input_rate * _CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * input_rate * _CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * output_rate
    ) / 1_000_000
    return cost


def write_state_snapshot(
    run_id: str, which: str, snapshot: dict[str, Any], runs_dir: Path | str = DEFAULT_RUNS_DIR
) -> Path:
    """state_before.json / state_after.json sit beside trace.json, not inside it -- they're a
    full DB snapshot (every table, every row), large enough that only the state-diff checker
    should ever have to load one. `which` is "before" or "after"."""
    assert which in ("before", "after"), which
    out_dir = Path(runs_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"state_{which}.json"
    path.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return path
