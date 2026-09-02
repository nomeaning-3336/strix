"""Tests for root supervision: per-child health tracking and supervision ticks."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from strix.core.agents import (
    AgentCoordinator,
    _tool_output_is_empty,
    action_fingerprint,
)
from strix.core.hooks import ReportUsageHooks, _result_looks_like_error
from strix.tools.agents_graph.tools import _wait_timeout_payload


def _make_coordinator() -> AgentCoordinator:
    coord = AgentCoordinator()
    coord.statuses = {"root": "waiting", "c1": "running", "c2": "running"}
    coord.parent_of = {"root": None, "c1": "root", "c2": "root"}
    coord.names = {"root": "Root Agent", "c1": "Hunter A", "c2": "Hunter B"}
    return coord


def _empty_exec_output() -> str:
    return "Chunk ID: abc123\nWall time: 0.1 seconds\nProcess exited with code 0\nOutput:\n"


def _exec_start(coord: AgentCoordinator, agent: str, cmd: str) -> None:
    coord.record_tool_start(agent, "exec_command", f"exec_command:{cmd}")


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


# --- action fingerprinting ---


def test_action_fingerprint_distinguishes_commands_on_same_tool() -> None:
    a = action_fingerprint("exec_command", {"cmd": "grep foo"})
    b = action_fingerprint("exec_command", {"cmd": "cat x"})
    c = action_fingerprint("exec_command", {"cmd": "grep foo"})
    assert a != b
    assert a == c


def test_action_fingerprint_is_argument_order_insensitive() -> None:
    left = action_fingerprint("exec_command", {"cmd": "ls", "tty": True})
    right = action_fingerprint("exec_command", {"tty": True, "cmd": "ls"})
    assert left == right


def test_action_fingerprint_handles_string_args_and_none() -> None:
    assert action_fingerprint("think", None) == "think"
    assert action_fingerprint("exec_command", '{"cmd": "echo hi"}').startswith("exec_command:")


# --- health tracking ---


def test_repeated_action_and_empty_output_tracking() -> None:
    coord = _make_coordinator()
    for _ in range(3):
        _exec_start(coord, "c1", "python3 -c 'print(1)'")
        coord.record_tool_end("c1", "exec_command", _empty_exec_output())
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["repeated_action_count"] == 3
    assert c1["empty_output_count"] == 3
    assert c1["seconds_since_progress"] is None


def test_different_commands_do_not_count_as_repeats() -> None:
    coord = _make_coordinator()
    for cmd in ("grep foo", "cat x", "npm test", "sed -n 1p"):
        _exec_start(coord, "c1", cmd)
        coord.record_tool_end("c1", "exec_command", "Chunk ID: x\nOutput:\nok\n")
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["repeated_action_count"] == 1


def test_identical_commands_increment_repeats() -> None:
    coord = _make_coordinator()
    for _ in range(4):
        _exec_start(coord, "c1", "python3 -c 'print(1)'")
        coord.record_tool_end("c1", "exec_command", "Chunk ID: x\nOutput:\nok\n")
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["repeated_action_count"] == 4


def test_progress_resets_empty_output_and_sets_stall_clock() -> None:
    coord = _make_coordinator()
    _exec_start(coord, "c1", "ls")
    coord.record_tool_end("c1", "exec_command", _empty_exec_output())
    _exec_start(coord, "c1", "ls /workspace")
    coord.record_tool_end("c1", "exec_command", "Chunk ID: y\nOutput:\nreal output\n")
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["empty_output_count"] == 0
    assert c1["seconds_since_progress"] == 0
    assert c1["last_progress_tool"] == "exec_command"


def test_error_results_increment_tool_errors() -> None:
    coord = _make_coordinator()
    _exec_start(coord, "c1", "ls")
    coord.record_tool_end(
        "c1", "exec_command", '{"success": false, "error": "boom"}', error=True
    )
    _exec_start(coord, "c1", "ls /workspace")
    coord.record_tool_end("c1", "exec_command", "Chunk ID: x\nOutput:\nok\n")
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["tool_errors"] == 1
    # an error is not also counted as empty-output; the later success is progress
    assert c1["empty_output_count"] == 0
    assert c1["seconds_since_progress"] == 0


def test_children_health_excludes_terminal_and_foreign_children() -> None:
    coord = _make_coordinator()
    coord.statuses["c2"] = "stopped"
    coord.parent_of["c3"] = "other"
    coord.statuses["c3"] = "running"
    assert [x["id"] for x in coord.children_health("root")] == ["c1"]


def test_llm_in_flight_distinguishes_reasoning_from_stall() -> None:
    coord = _make_coordinator()
    # progress long ago (looks stalled by tool clock alone)...
    _exec_start(coord, "c1", "ls")
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


# --- error-marker detection ---


def test_result_looks_like_error_markers() -> None:
    assert _result_looks_like_error('{"success": false, "error": "boom"}') is True
    assert _result_looks_like_error("error: something went wrong") is True
    assert _result_looks_like_error("Traceback (most recent call last):") is True
    exec_failed = "Chunk ID: x\nProcess exited with code 1\nOutput:\nno\n"
    assert _result_looks_like_error(exec_failed) is False
    assert _result_looks_like_error('{"success": true}') is False
    assert _result_looks_like_error("plain output containing error word") is False
    assert _result_looks_like_error(None) is False


# --- hook wiring ---


@pytest.mark.asyncio
async def test_hooks_feed_coordinator_health() -> None:
    coord = _make_coordinator()
    hooks = ReportUsageHooks(model="test-model")
    ctx = SimpleNamespace(
        context={"agent_id": "c1", "coordinator": coord},
        tool_arguments={"cmd": "python3 -c 'print(1)'"},
    )
    tool = SimpleNamespace(name="exec_command")
    await hooks.on_tool_start(ctx, MagicMock(), tool)
    await hooks.on_tool_end(ctx, MagicMock(), tool, _empty_exec_output())
    await hooks.on_tool_start(ctx, MagicMock(), tool)
    await hooks.on_tool_end(ctx, MagicMock(), tool, _empty_exec_output())
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["empty_output_count"] == 2
    assert c1["repeated_action_count"] == 2


@pytest.mark.asyncio
async def test_hooks_feed_tool_errors_from_error_results() -> None:
    coord = _make_coordinator()
    hooks = ReportUsageHooks(model="test-model")
    ctx = SimpleNamespace(
        context={"agent_id": "c1", "coordinator": coord},
        tool_arguments={"cmd": "ls"},
    )
    tool = SimpleNamespace(name="exec_command")
    await hooks.on_tool_start(ctx, MagicMock(), tool)
    await hooks.on_tool_end(ctx, MagicMock(), tool, '{"success": false, "error": "boom"}')
    c1 = next(x for x in coord.children_health("root") if x["id"] == "c1")
    assert c1["tool_errors"] == 1


@pytest.mark.asyncio
async def test_hooks_ignore_missing_coordinator() -> None:
    hooks = ReportUsageHooks(model="test-model")
    ctx = SimpleNamespace(context={"agent_id": "c1"})
    await hooks.on_tool_start(ctx, MagicMock(), SimpleNamespace(name="exec_command"))
    await hooks.on_tool_end(ctx, MagicMock(), SimpleNamespace(name="exec_command"), "x")


# --- supervision tick ---


def test_wait_timeout_returns_supervision_tick_when_children_exist() -> None:
    coord = _make_coordinator()
    _exec_start(coord, "c1", "python3 -c 'print(1)'")
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
