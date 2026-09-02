"""Tests for scan-agent tool registration in factory."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agents.tool import FunctionTool
from agents.tool_context import ToolContext

from strix.agents import factory
from strix.core.agents import AgentCoordinator
from strix.core.tool_policy import is_parallel_safe
from strix.tools.agents_graph.tools import wait_for_agents


def _tool(name: str) -> FunctionTool:
    # A per-tool closure keeps two same-named tools unequal, which is what the
    # duplicate-name tests exercise.
    async def invoke(_ctx: ToolContext[Any], _input: str) -> str:
        return "ok"

    return FunctionTool(
        name=name,
        description="test tool",
        params_json_schema={"type": "object", "properties": {}, "additionalProperties": False},
        on_invoke_tool=invoke,
    )


@pytest.fixture(autouse=True)
def _reset_registry() -> object:
    saved = list(factory._EXTRA_TOOLS)
    factory._EXTRA_TOOLS.clear()
    try:
        yield
    finally:
        factory._EXTRA_TOOLS[:] = saved


def test_register_agent_tools_is_deduped() -> None:
    tool = _tool("dup")
    factory.register_agent_tools(tool)
    factory.register_agent_tools(tool)
    assert factory.registered_agent_tools() == (tool,)


def test_registered_tools_appear_before_lifecycle_tool() -> None:
    tool = _tool("extra")
    factory.register_agent_tools(tool)

    root = factory.build_strix_agent(is_root=True)
    child = factory.build_strix_agent(is_root=False)

    root_names = [t.name for t in root.tools]
    child_names = [t.name for t in child.tools]

    assert root_names[-2:] == ["extra", "finish_scan"]
    assert child_names[-2:] == ["extra", "agent_finish"]


def test_per_call_extra_tools_stack_with_registry() -> None:
    factory.register_agent_tools(_tool("registered"))

    agent = factory.build_strix_agent(is_root=True, extra_tools=[_tool("per_call")])
    names = [t.name for t in agent.tools]

    assert "registered" in names
    assert "per_call" in names
    assert names[-1] == "finish_scan"


def test_register_agent_tools_rejects_duplicate_names() -> None:
    factory.register_agent_tools(_tool("same_name"))

    with pytest.raises(ValueError, match="same_name"):
        factory.register_agent_tools(_tool("same_name"))


def test_per_call_extra_tools_reject_duplicate_registered_names() -> None:
    factory.register_agent_tools(_tool("same_name"))

    with pytest.raises(ValueError, match="same_name"):
        factory.build_strix_agent(is_root=True, extra_tools=[_tool("same_name")])


def test_instructions_override_is_used_verbatim() -> None:
    custom = "You are a scan agent. Follow the provided scope."

    agent = factory.build_strix_agent(is_root=True, instructions_override=custom)

    assert agent.instructions == custom


def test_no_override_renders_builtin_prompt() -> None:
    agent = factory.build_strix_agent(is_root=True)

    assert isinstance(agent.instructions, str)
    assert agent.instructions != ""


def test_respond_to_user_is_interactive_only() -> None:
    """Yielding to the user is meaningless when no user is attached."""
    interactive = factory.build_strix_agent(is_root=True, interactive=True)
    autonomous = factory.build_strix_agent(is_root=True, interactive=False)

    assert "respond_to_user" in [t.name for t in interactive.tools]
    assert "respond_to_user" not in [t.name for t in autonomous.tools]


def test_wait_for_agents_is_available_in_both_modes() -> None:
    for interactive in (True, False):
        agent = factory.build_strix_agent(is_root=True, interactive=interactive)
        assert "wait_for_agents" in [t.name for t in agent.tools]


def test_source_inspect_many_is_registered_and_parallel_safe() -> None:
    root = factory.build_strix_agent(is_root=True)
    child = factory.build_strix_agent(is_root=False)
    for agent in (root, child):
        names = [t.name for t in agent.tools]
        assert "source_inspect_many" in names
    assert is_parallel_safe("source_inspect_many") is True


def test_repeated_builds_do_not_stack_execution_wrappers() -> None:
    """Regression: wrapping is idempotent.

    Every build_strix_agent wraps the module-singleton tools. Stacked wrappers
    would nest the width semaphore once per build and self-deadlock once the
    nesting depth exceeds the configured width (outer wrapper holds its permit
    while waiting for the inner chain), which hung wait_for_agents tests that
    ran after this file. The marker must make re-wrapping a no-op.
    """
    for _ in range(8):
        factory.build_strix_agent(is_root=True)

    async def _invoke() -> dict[str, object]:
        coordinator = AgentCoordinator()
        await coordinator.register("root", "strix", parent_id=None)
        ctx = ToolContext(
            context={"agent_id": "root", "coordinator": coordinator},
            tool_name="wait_for_agents",
            tool_call_id="call-1",
            tool_arguments="{}",
        )
        raw = await asyncio.wait_for(
            wait_for_agents.on_invoke_tool(
                ctx, json.dumps({"reason": "x", "timeout_seconds": 1})
            ),
            timeout=10,
        )
        return json.loads(raw)

    result = asyncio.run(_invoke())
    assert result["wait_outcome"] in {"timeout", "already_waited"}


def test_strict_tool_schemas_can_be_disabled_per_route() -> None:
    """Claude routes cap strict tools; the toolset must be sendable without strict."""
    agent = factory.build_strix_agent(is_root=True, strict_tool_schemas=False)

    function_tools = [t for t in agent.tools if isinstance(t, FunctionTool)]
    assert function_tools
    assert not any(t.strict_json_schema for t in function_tools)


def test_disabling_strict_leaves_shared_tools_untouched() -> None:
    factory.build_strix_agent(is_root=True, strict_tool_schemas=False)
    agent = factory.build_strix_agent(is_root=True)

    assert any(t.strict_json_schema for t in agent.tools if isinstance(t, FunctionTool))
