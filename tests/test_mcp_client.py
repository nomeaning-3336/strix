"""Tests for the generic MCP dispatch model.

Connections are connected without being registered as agent tools; their live
sessions go into a per-run registry; and every agent reaches them through the
two dispatch tools ``describe_mcp`` and ``call_mcp``.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import TYPE_CHECKING, Any

import pytest
from agents.mcp import MCPServer, MCPServerStdio, MCPServerStreamableHttp
from agents.tool_context import ToolContext
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from pydantic import ValidationError

from strix.agents import factory
from strix.interface.tui.live_view import TuiLiveView
from strix.tools.mcp import (
    MCP_REGISTRY_CONTEXT_KEY,
    BearerAuth,
    McpConnectionConfig,
    McpRegistry,
    call_mcp,
    describe_mcp,
    load_user_mcp_configs,
    mcp_inventory_context,
    namespaced_tool_name,
    resolve_mcp_tool,
)
from strix.tools.mcp import client as mcp_client


if TYPE_CHECKING:
    from pathlib import Path


class FakeMCPServer(MCPServer):
    """A connected MCP server stand-in, so tests never touch the network."""

    def __init__(self, name: str, tools: list[MCPTool]) -> None:
        super().__init__()
        self._name = name
        self._tools = tools
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    async def list_tools(
        self,
        run_context: Any = None,
        agent: Any = None,
    ) -> list[MCPTool]:
        return list(self._tools)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        self.calls.append((tool_name, arguments))
        return CallToolResult(content=[TextContent(type="text", text=f"routed:{tool_name}")])

    async def list_prompts(self) -> Any:
        raise NotImplementedError

    async def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError


class ErroringMCPServer(FakeMCPServer):
    """A connected server whose calls come back as MCP errors (isError=True)."""

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None,
        meta: dict[str, Any] | None = None,
    ) -> CallToolResult:
        self.calls.append((tool_name, arguments))
        return CallToolResult(
            content=[TextContent(type="text", text=f"boom:{tool_name}")],
            isError=True,
        )


def _mcp_tool(name: str, *, description: str | None = None) -> MCPTool:
    return MCPTool(
        name=name,
        description=description if description is not None else f"remote tool {name}",
        inputSchema={"type": "object", "properties": {"path": {"type": "string"}}},
    )


def _config(name: str, allowed_tools: list[str] | None) -> McpConnectionConfig:
    return McpConnectionConfig(
        name=name,
        url="https://mcp.example.com",
        auth=BearerAuth(token="run-token"),
        allowed_tools=allowed_tools,
    )


def _ctx(registry: McpRegistry | None) -> ToolContext[dict[str, Any]]:
    context: dict[str, Any] = {} if registry is None else {MCP_REGISTRY_CONTEXT_KEY: registry}
    return ToolContext(
        context=context,
        tool_name="mcp",
        tool_call_id="call-1",
        tool_arguments="{}",
    )


@pytest.fixture(autouse=True)
def _clear_mcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide any MCP settings the developer has exported in their own shell."""
    for name in ("STRIX_MCP_CONFIG", "STRIX_MCP_ONLY", "STRIX_MCP_EXCLUDE"):
        monkeypatch.delenv(name, raising=False)


# --- config contract ---------------------------------------------------------


def test_bearer_config_parses_from_dict() -> None:
    config = McpConnectionConfig.model_validate(
        {
            "name": "files_main",
            "transport": "http",
            "url": "https://mcp.example.com",
            "auth": {"kind": "bearer", "token": "abc"},
            "allowed_tools": ["list_files"],
        }
    )

    assert isinstance(config.auth, BearerAuth)
    assert config.auth.token == "abc"
    assert config.allowed_tools == ["list_files"]


def test_unknown_auth_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        McpConnectionConfig.model_validate(
            {
                "name": "x",
                "url": "https://mcp.example.com",
                "auth": {"kind": "oauth", "token": "abc"},
            }
        )


def test_stdio_config_parses_from_dict() -> None:
    config = McpConnectionConfig.model_validate(
        {
            "name": "local_fs",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"],
            "env": {"FOO": "bar"},
        }
    )

    assert config.transport == "stdio"
    assert config.command == "npx"
    assert config.args == ["-y", "@modelcontextprotocol/server-filesystem", "/srv/data"]
    assert config.env == {"FOO": "bar"}
    assert config.auth is None
    assert config.allowed_tools is None


def test_http_config_without_url_is_rejected() -> None:
    with pytest.raises(ValidationError):
        McpConnectionConfig.model_validate(
            {
                "name": "x",
                "transport": "http",
                "auth": {"kind": "bearer", "token": "abc"},
            }
        )


def test_stdio_config_without_command_is_rejected() -> None:
    with pytest.raises(ValidationError):
        McpConnectionConfig.model_validate({"name": "x", "transport": "stdio"})


def test_empty_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        McpConnectionConfig.model_validate(
            {
                "name": "",
                "url": "https://mcp.example.com",
                "auth": {"kind": "bearer", "token": "abc"},
            }
        )


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        McpConnectionConfig.model_validate(
            {
                "name": "x",
                "url": "https://mcp.example.com",
                "auth": {"kind": "bearer", "token": "abc"},
                "surprise": True,
            }
        )


# --- auth headers ------------------------------------------------------------


def test_bearer_auth_builds_authorization_header() -> None:
    headers = mcp_client._auth_headers(_config("files_main", []))

    assert headers == {"Authorization": "Bearer run-token"}


# --- connect without global registration -------------------------------------


@pytest.mark.asyncio
async def test_connect_returns_sessions_without_registering_agent_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = list(factory.registered_agent_tools())
    servers = {
        "fs": FakeMCPServer("fs", [_mcp_tool("read_file"), _mcp_tool("write_file")]),
        "db": FakeMCPServer("db", [_mcp_tool("query")]),
    }
    monkeypatch.setattr(mcp_client, "_build_server", lambda config: servers[config.name])

    connections = await mcp_client.connect_mcp_servers(
        [_config("fs", None), _config("db", ["query"])]
    )

    # The live sessions come back with their tool counts, and nothing was added
    # to the global agent-tool registry that pro shares.
    assert [(c.name, c.tool_count) for c in connections] == [("fs", 2), ("db", 1)]
    assert list(factory.registered_agent_tools()) == before


@pytest.mark.asyncio
async def test_tool_count_honors_the_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeMCPServer("fs", [_mcp_tool("read_file"), _mcp_tool("write_file")])
    monkeypatch.setattr(mcp_client, "_build_server", lambda _config: server)

    connections = await mcp_client.connect_mcp_servers([_config("fs", ["read_file"])])

    assert connections[0].tool_count == 1


@pytest.mark.asyncio
async def test_connection_notes_ride_on_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = FakeMCPServer("db", [_mcp_tool("query")])
    monkeypatch.setattr(mcp_client, "_build_server", lambda _config: server)
    config = McpConnectionConfig(
        name="db",
        url="https://mcp.example.com",
        notes="Staging analytics DB; read-only.",
        allowed_tools=["query"],
    )

    connections = await mcp_client.connect_mcp_servers([config])

    assert connections[0].notes == "Staging analytics DB; read-only."


# --- server build branch -----------------------------------------------------


def test_build_server_stdio_branch() -> None:
    config = McpConnectionConfig(
        name="local_fs",
        transport="stdio",
        command="my-server",
        args=["--flag", "value"],
        env={"TOKEN": "x"},
    )

    server = mcp_client._build_server(config)

    assert isinstance(server, MCPServerStdio)
    assert server.name == "local_fs"
    assert server.params.command == "my-server"
    assert server.params.args == ["--flag", "value"]
    assert server.params.env == {"TOKEN": "x"}


def test_build_server_http_branch() -> None:
    server = mcp_client._build_server(_config("files_main", ["list_files"]))

    assert isinstance(server, MCPServerStreamableHttp)
    assert server.name == "files_main"


# --- registry ----------------------------------------------------------------


def test_registry_add_get_and_names() -> None:
    registry = McpRegistry()
    server = FakeMCPServer("fs", [_mcp_tool("read_file")])

    registry.add(name="fs", server=server, purpose="local files", tool_count=1)

    entry = registry.get("fs")
    assert entry is not None
    assert entry.server is server
    assert entry.purpose == "local files"
    assert entry.tool_count == 1
    assert registry.get("missing") is None
    assert registry.names() == ["fs"]
    assert bool(registry) is True
    assert len(registry) == 1


def test_registry_summaries_and_inventory() -> None:
    registry = McpRegistry()
    registry.add(name="fs", server=FakeMCPServer("fs", []), purpose="local files", tool_count=2)
    registry.add(name="db", server=FakeMCPServer("db", []), purpose=None, tool_count=1)

    summaries = registry.summaries()
    assert [(s.name, s.purpose, s.tool_count) for s in summaries] == [
        ("fs", "local files", 2),
        ("db", None, 1),
    ]

    # The prompt inventory carries name/purpose/tool_count and no schemas.
    assert mcp_inventory_context(registry) == [
        {"name": "fs", "purpose": "local files", "tool_count": 2},
        {"name": "db", "purpose": None, "tool_count": 1},
    ]


def test_inventory_is_empty_without_a_registry() -> None:
    assert mcp_inventory_context(None) == []
    assert mcp_inventory_context(McpRegistry()) == []


# --- describe_mcp ------------------------------------------------------------


@pytest.mark.asyncio
async def test_describe_mcp_returns_tool_names_and_schemas() -> None:
    registry = McpRegistry()
    server = FakeMCPServer("fs", [_mcp_tool("read_file", description="Read a file")])
    registry.add(name="fs", server=server, purpose=None, tool_count=1)

    out = await describe_mcp.on_invoke_tool(_ctx(registry), json.dumps({"connection": "fs"}))

    assert "read_file" in out
    assert "Read a file" in out
    # The tool's JSON input schema is shown so the model can build call arguments.
    assert '"path"' in out


@pytest.mark.asyncio
async def test_describe_mcp_errors_clearly_on_unknown_connection() -> None:
    registry = McpRegistry()
    registry.add(name="fs", server=FakeMCPServer("fs", []), purpose=None, tool_count=0)

    out = await describe_mcp.on_invoke_tool(_ctx(registry), json.dumps({"connection": "nope"}))

    assert "Unknown MCP connection 'nope'" in out
    assert "fs" in out


@pytest.mark.asyncio
async def test_describe_mcp_without_any_connections() -> None:
    out = await describe_mcp.on_invoke_tool(_ctx(None), json.dumps({"connection": "fs"}))

    assert out == "No MCP connections are configured for this run."


# --- call_mcp ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_mcp_dispatches_and_returns_converted_output() -> None:
    registry = McpRegistry()
    server = FakeMCPServer("fs", [_mcp_tool("read_file")])
    registry.add(name="fs", server=server, purpose=None, tool_count=1)

    out = await call_mcp.on_invoke_tool(
        _ctx(registry),
        json.dumps({"connection": "fs", "tool": "read_file", "arguments": {"path": "/etc/hosts"}}),
    )

    # The call reaches the server by the unprefixed tool name with its arguments.
    assert server.calls == [("read_file", {"path": "/etc/hosts"})]
    assert out == {"type": "text", "text": "routed:read_file"}


@pytest.mark.asyncio
async def test_call_mcp_defaults_missing_arguments_to_empty_object() -> None:
    registry = McpRegistry()
    server = FakeMCPServer("fs", [_mcp_tool("ping")])
    registry.add(name="fs", server=server, purpose=None, tool_count=1)

    await call_mcp.on_invoke_tool(_ctx(registry), json.dumps({"connection": "fs", "tool": "ping"}))

    assert server.calls == [("ping", {})]


@pytest.mark.asyncio
async def test_call_mcp_errors_on_unknown_connection() -> None:
    registry = McpRegistry()
    registry.add(name="fs", server=FakeMCPServer("fs", []), purpose=None, tool_count=0)

    out = await call_mcp.on_invoke_tool(
        _ctx(registry), json.dumps({"connection": "nope", "tool": "x"})
    )

    assert "Unknown MCP connection 'nope'" in out
    assert "fs" in out


@pytest.mark.asyncio
async def test_call_mcp_errors_on_unknown_tool() -> None:
    registry = McpRegistry()
    server = FakeMCPServer("fs", [_mcp_tool("read_file")])
    registry.add(name="fs", server=server, purpose=None, tool_count=1)

    out = await call_mcp.on_invoke_tool(
        _ctx(registry), json.dumps({"connection": "fs", "tool": "delete_everything"})
    )

    assert "Unknown tool 'delete_everything'" in out
    assert "read_file" in out
    # A rejected tool name never reaches the server.
    assert server.calls == []


@pytest.mark.asyncio
async def test_call_mcp_errors_on_non_dict_arguments() -> None:
    registry = McpRegistry()
    server = FakeMCPServer("fs", [_mcp_tool("read_file")])
    registry.add(name="fs", server=server, purpose=None, tool_count=1)

    out = await call_mcp.on_invoke_tool(
        _ctx(registry),
        json.dumps({"connection": "fs", "tool": "read_file", "arguments": ["not", "a", "dict"]}),
    )

    assert "expected a JSON object" in out
    assert server.calls == []


@pytest.mark.asyncio
async def test_call_mcp_applies_a_connection_result_transform() -> None:
    registry = McpRegistry()
    server = FakeMCPServer("fs", [_mcp_tool("read_file")])
    seen: list[tuple[str, Any]] = []

    def transform(label: str, structured: Any) -> Any:
        seen.append((label, structured))
        return {"kept": structured["content"][0]["text"]}

    registry.add(name="fs", server=server, purpose=None, tool_count=1, result_transform=transform)

    out = await call_mcp.on_invoke_tool(
        _ctx(registry), json.dumps({"connection": "fs", "tool": "read_file"})
    )

    # The transform sees the model-facing <connection>_<tool> label and the
    # parsed CallToolResult, and its return becomes the tool output.
    assert seen[0][0] == "fs_read_file"
    assert seen[0][1]["content"][0]["text"] == "routed:read_file"
    assert out == {"kept": "routed:read_file"}


@pytest.mark.asyncio
async def test_call_mcp_flags_an_errored_result_failed_for_the_tui() -> None:
    registry = McpRegistry()
    server = ErroringMCPServer("fs", [_mcp_tool("read_file")])
    registry.add(name="fs", server=server, purpose=None, tool_count=1)

    out = await call_mcp.on_invoke_tool(
        _ctx(registry), json.dumps({"connection": "fs", "tool": "read_file"})
    )

    # The agent content is unchanged; success:False rides alongside so the TUI
    # can tell an errored call from a done one.
    assert out == {"type": "text", "text": "boom:read_file", "success": False}


# --- the two tools are the only MCP surface every agent gets -----------------


def test_agent_carries_exactly_the_two_dispatch_tools_regardless_of_connections() -> None:
    """No matter how many MCP connections a run makes, an agent's tool list gains
    exactly describe_mcp and call_mcp and never a per-connection provider tool."""
    root = factory.build_strix_agent(is_root=True)
    child = factory.build_strix_agent(is_root=False)

    root_names = [t.name for t in root.tools]
    child_names = [t.name for t in child.tools]

    assert {"describe_mcp", "call_mcp"} <= set(root_names)
    assert {"describe_mcp", "call_mcp"} <= set(child_names)

    # Five hypothetical connections would once have added ~all their tools as
    # namespaced provider tools; none of those names may appear now.
    provider_names = {
        namespaced_tool_name(f"conn{i}", tool)
        for i in range(5)
        for tool in ("read_file", "write_file", "query")
    }
    assert provider_names.isdisjoint(root_names)
    assert provider_names.isdisjoint(child_names)

    # The tool list does not grow with connection count: it is the same set of
    # names whether or not any connection exists, because connections never
    # contribute tools.
    assert root_names == [t.name for t in factory.build_strix_agent(is_root=True).tools]


# --- loader ------------------------------------------------------------------


def test_loader_parses_stdio_and_http_entries(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp-servers.json"
    config_file.write_text(
        json.dumps(
            [
                {
                    "name": "local_fs",
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "server-filesystem"],
                },
                {
                    "name": "files_main",
                    "transport": "http",
                    "url": "https://mcp.example.com",
                    "auth": {"kind": "bearer", "token": "abc"},
                    "allowed_tools": ["list_files"],
                },
            ]
        ),
        encoding="utf-8",
    )

    configs = load_user_mcp_configs(config_file)

    assert [c.name for c in configs] == ["local_fs", "files_main"]
    assert configs[0].transport == "stdio"
    assert configs[1].allowed_tools == ["list_files"]


def test_loader_skips_bad_entry_but_keeps_good_ones(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp-servers.json"
    config_file.write_text(
        json.dumps(
            [
                {"name": "broken", "transport": "http"},
                {"name": "local_fs", "transport": "stdio", "command": "npx"},
            ]
        ),
        encoding="utf-8",
    )

    configs = load_user_mcp_configs(config_file)

    assert [c.name for c in configs] == ["local_fs"]


def test_loader_returns_empty_when_file_absent(tmp_path: Path) -> None:
    assert load_user_mcp_configs(tmp_path / "does-not-exist.json") == []


def test_loader_reads_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_file = tmp_path / "from-env.json"
    config_file.write_text(
        json.dumps([{"name": "local_fs", "transport": "stdio", "command": "npx"}]),
        encoding="utf-8",
    )
    monkeypatch.setenv("STRIX_MCP_CONFIG", str(config_file))

    configs = load_user_mcp_configs()

    assert [c.name for c in configs] == ["local_fs"]


def _names_file(tmp_path: Path, *names: str) -> Path:
    config_file = tmp_path / "mcp-servers.json"
    config_file.write_text(
        json.dumps([{"name": n, "transport": "stdio", "command": "npx"} for n in names]),
        encoding="utf-8",
    )
    return config_file


def test_loader_drops_duplicate_named_connections(tmp_path: Path) -> None:
    config_file = tmp_path / "mcp-servers.json"
    config_file.write_text(
        json.dumps(
            [
                {"name": "dup", "transport": "stdio", "command": "first"},
                {"name": "dup", "transport": "stdio", "command": "second"},
                {"name": "other", "transport": "stdio", "command": "npx"},
            ]
        ),
        encoding="utf-8",
    )

    configs = load_user_mcp_configs(config_file)

    assert [c.name for c in configs] == ["dup", "other"]
    assert configs[0].command == "first"


def test_loader_include_selection_keeps_only_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = _names_file(tmp_path, "a", "b", "c")
    monkeypatch.setenv("STRIX_MCP_ONLY", "a,c")

    configs = load_user_mcp_configs(config_file)

    assert [c.name for c in configs] == ["a", "c"]


def test_loader_exclude_selection_drops_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = _names_file(tmp_path, "a", "b", "c")
    monkeypatch.setenv("STRIX_MCP_EXCLUDE", "b")

    configs = load_user_mcp_configs(config_file)

    assert [c.name for c in configs] == ["a", "c"]


# --- cancellation cleanup ----------------------------------------------------


@pytest.mark.asyncio
async def test_connect_cleans_up_when_cancelled_mid_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned: list[str] = []

    class _Tracking(FakeMCPServer):
        def __init__(self, name: str, *, fail_connect: bool = False) -> None:
            super().__init__(name, [_mcp_tool("t")])
            self._fail_connect = fail_connect

        async def connect(self) -> None:
            if self._fail_connect:
                raise asyncio.CancelledError

        async def cleanup(self) -> None:
            cleaned.append(self._name)

    servers = {"good": _Tracking("good"), "bad": _Tracking("bad", fail_connect=True)}
    monkeypatch.setattr(mcp_client, "_build_server", lambda config: servers[config.name])

    configs = [
        McpConnectionConfig(name="good", url="https://mcp.example.com", allowed_tools=["t"]),
        McpConnectionConfig(name="bad", url="https://mcp.example.com", allowed_tools=["t"]),
    ]

    with pytest.raises(asyncio.CancelledError):
        await mcp_client.connect_mcp_servers(configs)

    # The server being connected when cancelled, and the one already connected,
    # are both cleaned up rather than orphaned.
    assert cleaned == ["bad", "good"]


# --- reading a tool call back to the server it went out to -------------------
# resolve_mcp_tool / namespaced_tool_name stay in strix.tools.mcp.naming: the
# TUI reads them to attribute a call to its connection, and call_mcp builds the
# result_transform label with namespaced_tool_name.


def test_resolve_mcp_tool_splits_against_the_run_connections() -> None:
    assert resolve_mcp_tool("local_fs_read_file", ["github", "local_fs"]) == (
        "local_fs",
        "read_file",
    )


def test_resolve_mcp_tool_prefers_the_longest_matching_connection() -> None:
    assert resolve_mcp_tool("files_main_list", ["files", "files_main"]) == ("files_main", "list")


def test_resolve_mcp_tool_matches_a_connection_name_it_had_to_sanitize() -> None:
    tool_name = namespaced_tool_name("my server", "db.query")

    assert resolve_mcp_tool(tool_name, ["my server"]) == ("my server", "db_query")


def test_resolve_mcp_tool_ignores_tools_that_are_not_a_connection_s() -> None:
    assert resolve_mcp_tool("exec_command", ["local_fs"]) is None
    assert resolve_mcp_tool("local_fsx", ["local_fs"]) is None


def test_namespaced_name_is_a_valid_tool_name() -> None:
    # A connection named with a space and a server tool named with a dot still
    # sanitize to a valid model-facing label for the result_transform.
    name = namespaced_tool_name("my server", "db.query")

    assert name == "my_server_db_query"
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", name)


def test_projected_tool_call_names_the_server_it_went_out_to() -> None:
    view = TuiLiveView()
    view.set_mcp_connections(["local_fs"])

    view._record_tool_call_data(
        "agent-1",
        {"call_id": "c1", "tool_name": "local_fs_read_file", "args": {"path": "/etc/hosts"}},
    )
    view._record_tool_call_data(
        "agent-1",
        {"call_id": "c2", "tool_name": "exec_command", "args": {"cmd": "ls"}},
    )

    mcp_call, built_in = (event["data"] for event in view.events)
    assert (mcp_call["mcp_connection"], mcp_call["mcp_tool"]) == ("local_fs", "read_file")
    assert "mcp_connection" not in built_in
