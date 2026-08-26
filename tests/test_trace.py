"""agent/trace.py: the trace record shape, its JSON writer, and cost computation from the four
usage counters. No API calls here -- this is pure data plumbing."""

from __future__ import annotations

import json

from agent.trace import (
    Trace,
    ToolCallRecord,
    TurnRecord,
    UsageRecord,
    compute_cost_usd,
    write_state_snapshot,
)


def _make_trace(**overrides) -> Trace:
    defaults = dict(
        run_id="run-1", case_id="hp_01_book_standard_C", persona="C", variant="baseline",
        model="claude-sonnet-5", effort="high", replicate=0,
        principal={"type": "customer", "id": 14},
    )
    defaults.update(overrides)
    return Trace(**defaults)


def test_trace_round_trips_through_json(tmp_path):
    trace = _make_trace()
    trace.turns.append(TurnRecord(role="user", text="Book me a drain cleaning."))
    trace.turns.append(TurnRecord(
        role="assistant", text="Sure, let me check.",
        tool_calls=[ToolCallRecord(
            tool="book_appointment", args={"service_item_id": 2}, declared_tier=1,
            result={"decision": "executed", "appointment_id": 5}, decision="executed",
            reason=None, entity_ref="appointment:5", latency_ms=812.3,
        )],
    ))
    trace.usage = UsageRecord(input_tokens=100, output_tokens=50, cost_usd=0.001)
    path = trace.write(runs_dir=tmp_path)

    assert path == tmp_path / "run-1" / "trace.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "run-1"
    assert loaded["turns"][1]["tool_calls"][0]["tool"] == "book_appointment"
    assert loaded["turns"][1]["tool_calls"][0]["decision"] == "executed"
    assert loaded["usage"]["input_tokens"] == 100


def test_trace_defaults_are_sane():
    trace = _make_trace()
    assert trace.turns == []
    assert trace.hit_turn_cap is False
    assert trace.outcome == "ok"
    assert trace.error_detail is None
    assert trace.usage.cost_usd == 0.0


def test_run_dir_matches_write_location(tmp_path):
    trace = _make_trace()
    assert trace.run_dir(tmp_path) == tmp_path / "run-1"
    trace.write(runs_dir=tmp_path)
    assert (trace.run_dir(tmp_path) / "trace.json").exists()


def test_cost_computed_from_all_four_usage_counters():
    usage = UsageRecord(
        input_tokens=1_000_000, cache_read_input_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000, output_tokens=1_000_000,
    )
    cost = compute_cost_usd("claude-sonnet-5", usage)
    # sonnet-5 intro rate: $2 input, $10 output per MTok; cache read 0.1x, cache write 1.25x
    expected = 2.00 + (2.00 * 0.1) + (2.00 * 1.25) + 10.00
    assert cost == expected


def test_cost_from_input_tokens_alone_is_not_free():
    """Costing from input_tokens alone would make a heavily-cached run look free. A run with
    only cache reads must still show nonzero cost."""
    usage = UsageRecord(cache_read_input_tokens=1_000_000)
    cost = compute_cost_usd("claude-sonnet-5", usage)
    assert cost > 0


def test_unknown_model_returns_zero_cost_not_a_guess():
    usage = UsageRecord(input_tokens=1_000_000, output_tokens=1_000_000)
    assert compute_cost_usd("some-future-model", usage) == 0.0


def test_state_snapshot_written_beside_trace_not_inside_it(tmp_path):
    trace = _make_trace()
    trace_path = trace.write(runs_dir=tmp_path)
    before_path = write_state_snapshot("run-1", "before", {"customers": []}, runs_dir=tmp_path)
    after_path = write_state_snapshot("run-1", "after", {"customers": [{"id": 1}]}, runs_dir=tmp_path)

    assert before_path.parent == trace_path.parent
    assert before_path.name == "state_before.json"
    assert after_path.name == "state_after.json"
    loaded_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "customers" not in loaded_trace  # state never gets embedded in the trace itself
    assert json.loads(after_path.read_text(encoding="utf-8"))["customers"] == [{"id": 1}]
