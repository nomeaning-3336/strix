"""Tests for root supervision: per-child health tracking and supervision ticks."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from strix.core.agents import AgentCoordinator, _tool_output_is_empty
from strix.core.hooks import ReportUsageHooks
from strix.tools.agents_graph.tools import _wait_timeout_payload


def _make_coordinator() -> AgentCoordinator:
    coord = AgentCoordinator()
    coord.statuses = {"root": "waiting", "c1": "running", "c2": "running"}
    coord.parent_of = {"root": None, "c1": "root", "c2": "root"}
    coord.names = {"root": "Root Agent", "c1": "Hunter A", "c2": "Hunter B"}
    return coord


def _empty_exec_output() -> str:
    return "Chunk ID: abc123\nWall time: 0.1 seconds\nProcess exited with code 0\nOutput:\n"


# --- empty-output detection ---


def test_tool_output_is_empty_detects_empty_exec_stdout() -> None:
    assert _tool_output_is_empty("exec_command", _empty_exec_output()) is True
    assert _tool_output_is_empty("write_stdin", _empty_exec_output()) is True


def test_tool_output_is_empty_accepts_real_stdout() -> None:
    assert _tool_output_is_empty("exec_command", "Chunk ID: x\nOutput:\nreal stdout\n") is False


def test_tool_output_is_empty_none_and_blank() -> None:
    assert _tool_output_is_empty("exec_command", None) is True
    assert _tool_output_is_empty("read_file", "   ") is True
    assert _tool_output_is_empty("read_file", "content") is False


# --- health tracking ---


def test_repeated_action_and_empty_output_tracking() -> None:
    coord = _make_coordinator()
    for _ in range(3):
        coord.record_tool_start("c1", "exec_command")
        coord.record_tool_end("c1", "exec_command", _empty_exec_output())
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["repeated_action_count"] == 3
    assert c1["empty_output_count"] == 3
    assert c1["seconds_since_progress"] is None


def test_progress_resets_empty_output_and_sets_stall_clock() -> None:
    coord = _make_coordinator()
    coord.record_tool_start("c1", "exec_command")
    coord.record_tool_end("c1", "exec_command", _empty_exec_output())
    coord.record_tool_start("c1", "exec_command")
    coord.record_tool_end("c1", "exec_command", "Chunk ID: y\nOutput:\nreal output\n")
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["empty_output_count"] == 0
    assert c1["seconds_since_progress"] == 0
    assert c1["last_progress_tool"] == "exec_command"


def test_children_health_excludes_terminal_and_foreign_children() -> None:
    coord = _make_coordinator()
    coord.statuses["c2"] = "stopped"
    coord.parent_of["c3"] = "other"
    coord.statuses["c3"] = "running"
    assert [x["id"] for x in coord.children_health("root")] == ["c1"]


def test_llm_in_flight_distinguishes_reasoning_from_stall() -> None:
    coord = _make_coordinator()
    # progress long ago (looks stalled by tool clock alone)...
    coord.record_tool_start("c1", "exec_command")
    coord.record_tool_end("c1", "exec_command", "Chunk ID: x\nOutput:\nreal\n")
    # ...but the model is mid-turn: in flight must read as working
    coord.mark_llm_start("c1")
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["llm_in_flight"] is True
    assert c1["in_flight_seconds"] == 0

    coord.mark_llm_end("c1")
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["llm_in_flight"] is False
    assert c1["in_flight_seconds"] is None


def test_llm_in_flight_defaults_false_without_model_activity() -> None:
    coord = _make_coordinator()
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["llm_in_flight"] is False
    assert c1["in_flight_seconds"] is None


# --- hook wiring ---


@pytest.mark.asyncio
async def test_hooks_feed_coordinator_health() -> None:
    coord = _make_coordinator()
    hooks = ReportUsageHooks(model="test-model")
    ctx = SimpleNamespace(context={"agent_id": "c1", "coordinator": coord})
    tool = SimpleNamespace(name="exec_command")
    await hooks.on_tool_start(ctx, MagicMock(), tool)
    await hooks.on_tool_end(ctx, MagicMock(), tool, _empty_exec_output())
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["empty_output_count"] == 1
    assert c1["repeated_action_count"] == 1


@pytest.mark.asyncio
async def test_hooks_ignore_missing_coordinator() -> None:
    hooks = ReportUsageHooks(model="test-model")
    ctx = SimpleNamespace(context={"agent_id": "c1"})
    await hooks.on_tool_start(ctx, MagicMock(), SimpleNamespace(name="exec_command"))
    await hooks.on_tool_end(ctx, MagicMock(), SimpleNamespace(name="exec_command"), "x")


# --- supervision tick ---


def test_wait_timeout_returns_supervision_tick_when_children_exist() -> None:
    coord = _make_coordinator()
    coord.record_tool_start("c1", "exec_command")
    coord.record_tool_end("c1", "exec_command", _empty_exec_output())

    data = json.loads(_wait_timeout_payload(coord, "root", 45, "supervise"))

    assert data["wait_outcome"] == "supervision_tick"
    assert any(c["id"] == "c1" and c["repeated_action_count"] == 1 for c in data["children"])


def test_wait_timeout_returns_timeout_not_tick_when_no_children() -> None:
    coord = _make_coordinator()
    coord.parent_of = {"root": None}  # no children

    data = json.loads(_wait_timeout_payload(coord, "root", 45, "s"))

    assert data["wait_outcome"] == "timeout"
    assert "children" not in data
