"""Wide-turn execution (Efficiency v1): policy, parallel enablement, telemetry.

Covers the harness-side guarantees that make provider ``parallel_tool_calls``
safe and measurable:

  - ToolPolicy: which tools may batch (read-only) vs must stay serial.
  - serialized(): same-loop concurrent calls of one serial group never overlap.
  - make_model_settings parallel resolution (auto/off/override, tool-less None).
  - AgentCoordinator wide-turn counters (model_requests, tool groups, widths,
    serial-equivalent vs wall time) and per-agent cache token recording.
  - cached_input_tokens parsing + the run-record cache fields.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from strix.core import inputs
from strix.core.agents import AgentCoordinator
from strix.core.inputs import make_model_settings
from strix.core.tool_policy import (
    DEFAULT_POLICY,
    is_parallel_safe,
    parallel_safe_tool_names,
    policy_for,
    serial_group_for,
    serialized,
    wide_turn_guidance,
)
from strix.report.usage import LLMUsageLedger, cached_input_tokens


if TYPE_CHECKING:

    import pytest


def _settings_with(parallel_tool_calls: bool | None) -> SimpleNamespace:
    return SimpleNamespace(llm=SimpleNamespace(parallel_tool_calls=parallel_tool_calls))


# --- tool policy -------------------------------------------------------------


def test_read_only_reader_tools_are_parallel_safe() -> None:
    for name in (
        "list_reports",
        "get_report",
        "list_todos",
        "get_threat_model",
        "view_agent_graph",
    ):
        policy = policy_for(name)
        assert policy.parallel_safe is True
        assert policy.side_effect_free is True
        assert policy.cacheable is True
    assert policy_for("source_inspect_many").parallel_safe is True


def test_shell_and_unknown_tools_stay_serial() -> None:
    assert policy_for("exec_command").parallel_safe is False
    assert policy_for("write_stdin").parallel_safe is False
    assert serial_group_for("exec_command") == serial_group_for("write_stdin") == "shell"
    # Mutating / arbitrary tools default to fully serial, serialized alone.
    assert policy_for("create_vulnerability_report") == DEFAULT_POLICY
    assert serial_group_for("create_vulnerability_report") == "create_vulnerability_report"
    assert policy_for("some_unknown_tool").parallel_safe is False
    assert is_parallel_safe(None) is False


def test_guidance_names_only_safe_tools_and_width() -> None:
    safe = set(parallel_safe_tool_names())
    guidance = wide_turn_guidance(3)
    assert "up to 3 independent tool calls" in guidance
    assert "exec_command" not in safe
    for name in list(safe)[:3]:
        assert name in guidance
    assert "NEVER issue exec_command/write_stdin" in guidance
    # Deterministic ordering for a stable prompt prefix.
    assert parallel_safe_tool_names() == sorted(parallel_safe_tool_names())


async def test_serialized_prevents_overlap_within_loop() -> None:
    active = 0
    peak = 0
    lock_state = threading.Lock()

    async def worker() -> None:
        nonlocal active, peak
        async with serialized("shared"):
            with lock_state:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.03)
            with lock_state:
                active -= 1

    await asyncio.gather(*(worker() for _ in range(4)))
    assert peak == 1


async def test_different_serial_groups_may_overlap() -> None:
    async def worker(group: str, track: list[float]) -> None:
        start = time.monotonic()
        async with serialized(group):
            await asyncio.sleep(0.06)
        track.append(time.monotonic() - start)

    timings: list[float] = []
    await asyncio.gather(worker("a", timings), worker("b", timings))
    # Both finished in ~one sleep period, not two serialized ones.
    assert max(timings) < 0.11


# --- parallel enablement in model settings ------------------------------------


def test_make_model_settings_parallel_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def with_parallel(value: bool | None) -> Any:
        monkeypatch.setattr(inputs, "load_settings", lambda: _settings_with(value))
        return make_model_settings("medium", model_name="openrouter/model-x")

    monkeypatch.setattr(inputs, "load_settings", lambda: _settings_with(None))
    assert with_parallel(value=None).parallel_tool_calls is True  # auto-enable
    assert with_parallel(value=False).parallel_tool_calls is False
    assert with_parallel(value=True).parallel_tool_calls is True

    # Tool-less calls (dedupe, compaction, warm-up) never set parallel.
    monkeypatch.setattr(inputs, "load_settings", lambda: _settings_with(None))
    model_settings = make_model_settings(
        "medium", model_name="openrouter/model-x", has_tools=False
    )
    assert model_settings.parallel_tool_calls is None


# --- wide-turn counters on the coordinator ------------------------------------


def test_coordinator_wide_turn_counters() -> None:
    coordinator = AgentCoordinator()
    agent_id = "agent-1"

    # Turn 1: two concurrent tool calls in one group.
    coordinator.mark_llm_start(agent_id)
    coordinator.record_tool_start(agent_id, "exec_command", "exec_command:grep a")
    coordinator.record_tool_start(agent_id, "list_reports", "list_reports:{}")
    time.sleep(0.002)
    coordinator.record_tool_end(agent_id, "list_reports", "ok")
    coordinator.record_tool_end(agent_id, "exec_command", "ok")
    coordinator.mark_llm_end(agent_id)

    # Turn 2: pure reasoning, no tools.
    coordinator.mark_llm_start(agent_id)
    coordinator.mark_llm_end(agent_id)

    coordinator.record_llm_usage(
        agent_id, input_tokens=100, cached_input_tokens=90, output_tokens=10
    )

    health = coordinator.health[agent_id]
    assert health.model_requests == 2
    assert health.tool_calls == 2
    assert health.tool_groups == 1  # turn 2 had no tools, contributes no group
    assert health.width_sum == 2
    assert health.tools_serial_ms > 0
    assert health.tools_wall_ms > 0
    assert health.input_tokens == 100
    assert health.cached_input_tokens == 90
    assert health.output_tokens == 10

    counters = coordinator._counters(health)
    assert counters["avg_tool_width"] == 2.0
    assert counters["model_requests"] == 2
    assert counters["cache_ratio"] == 0.9
    assert counters["tools_serial_ms"] > 0
    assert counters["tools_wall_ms"] > 0


def test_children_health_carries_wide_turn_metrics() -> None:
    coordinator = AgentCoordinator()
    coordinator.statuses["child"] = "running"
    coordinator.parent_of["child"] = "root"
    coordinator.names["child"] = "Worker"
    coordinator.record_tool_start("child", "read_file", "read_file:{}")
    coordinator.record_tool_end("child", "read_file", "content")
    coordinator.mark_llm_end("child")

    snapshot = coordinator.children_health("root")
    assert len(snapshot) == 1
    entry = snapshot[0]
    assert entry["model_requests"] == 1
    assert entry["tool_calls"] == 1
    assert entry["avg_tool_width"] == 1.0
    assert "cache_ratio" in entry

    # runtime_snapshot exposes the same counters for the live viewer.
    runtime = coordinator.runtime_snapshot()
    assert runtime["child"]["tool_groups"] == 1


def test_health_empty_counters_are_stable() -> None:
    coordinator = AgentCoordinator()
    empty = coordinator._counters(None)
    assert empty["avg_tool_width"] is None
    assert empty["tools_serial_ms"] == 0
    assert empty["cache_ratio"] is None
    assert set(empty) == {
        "model_requests",
        "tool_calls",
        "tool_groups",
        "avg_tool_width",
        "tools_serial_ms",
        "tools_wall_ms",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "cache_ratio",
    }


# --- cache token accounting ---------------------------------------------------


def test_cached_input_tokens_parsing() -> None:
    entry = SimpleNamespace(input_tokens_details={"cached_tokens": 42})
    usage_with_entries = SimpleNamespace(request_usage_entries=[entry], input_tokens_details=None)
    assert cached_input_tokens(usage_with_entries) == 42

    top_level = SimpleNamespace(
        request_usage_entries=[],
        input_tokens_details={"cached_tokens": 7},
    )
    assert cached_input_tokens(top_level) == 7

    assert cached_input_tokens(None) == 0
    assert cached_input_tokens(SimpleNamespace(request_usage_entries=None)) == 0


def test_ledger_record_exposes_cache_fields() -> None:
    ledger = LLMUsageLedger()
    ledger._cached_input = 120
    ledger._uncached_input = 80
    record = ledger.to_record()
    assert record["cached_input_tokens"] == 120
    assert record["uncached_input_tokens"] == 80
    assert record["cache_ratio"] == 0.6
