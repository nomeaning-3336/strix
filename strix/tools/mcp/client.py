"""Connect to MCP servers so a run can reach their tools on demand.

Given one :class:`McpConnectionConfig` per server, :func:`connect_mcp_servers`
connects each server, counts the tools it offers (honoring the connection's
allowlist), and returns the live sessions. It does NOT register anything as an
agent tool: under the generic-dispatch model the run holds these sessions in a
per-run :class:`~strix.tools.mcp.registry.McpRegistry`, and the agent reaches
them through the two dispatch tools (``describe_mcp`` / ``call_mcp``), which call
:func:`dispatch_mcp_call` here to run one tool and serialize its result.

A server that cannot connect is logged and skipped, so one bad connection never
fails the run.
"""

from __future__ import annotations

import contextlib
import json
import logging
from typing import TYPE_CHECKING, Any, NamedTuple, cast

from agents.mcp import (
    MCPServer,
    MCPServerStdio,
    MCPServerStdioParams,
    MCPServerStreamableHttp,
    MCPServerStreamableHttpParams,
    create_static_tool_filter,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from strix.tools.mcp.config import McpConnectionConfig

    # Runs on one tool call's structured result before it reaches the agent.
    # Called ``result_transform(label, structured_result)`` and its return value
    # becomes the tool's output. ``label`` is the model-facing
    # ``<connection>_<tool>`` name so a transform keyed on names still resolves
    # the same way it did under per-tool registration; ``structured_result`` is
    # the parsed ``CallToolResult`` as a dict (not a serialized string), so the
    # transform can project or drop individual fields.
    ResultTransform = Callable[[str, Any], Any]


logger = logging.getLogger(__name__)


class ConnectedMcpServer(NamedTuple):
    """One successfully connected MCP server and how many tools it offers.

    ``server`` is kept so the caller can clean it up when the run ends, and so
    the caller can hand the live session to the run's
    :class:`~strix.tools.mcp.registry.McpRegistry`; ``name`` and ``tool_count``
    let the caller show the user a startup summary and fill the prompt inventory;
    ``notes`` carries the connection's optional free-text description so the
    caller can surface it as the connection's purpose in the inventory.
    """

    server: MCPServer
    name: str
    tool_count: int
    notes: str | None = None


def _auth_headers(config: McpConnectionConfig) -> dict[str, str]:
    """Build the per-server request headers from the connection's auth."""
    auth = config.auth
    if auth is None:
        return {}
    return {"Authorization": f"Bearer {auth.token}"}


def _build_server(config: McpConnectionConfig) -> MCPServer:
    """Construct (but do not connect) the SDK server for one connection.

    When ``allowed_tools`` is a list the static filter means the server will not
    even list tools outside it, so it is the authoritative gate on what
    ``describe_mcp`` and ``call_mcp`` can see. When it is ``None`` no filter is
    applied and every listed tool is reachable.
    """
    tool_filter = (
        create_static_tool_filter(allowed_tool_names=config.allowed_tools)
        if config.allowed_tools is not None
        else None
    )

    if config.transport == "stdio":
        stdio_params: MCPServerStdioParams = {
            "command": cast("str", config.command),
            "args": config.args,
            "env": config.env,
        }
        return MCPServerStdio(
            params=stdio_params,
            name=config.name,
            tool_filter=tool_filter,
            cache_tools_list=True,
        )

    http_params: MCPServerStreamableHttpParams = {
        "url": cast("str", config.url),
        "headers": _auth_headers(config),
    }
    return MCPServerStreamableHttp(
        params=http_params,
        name=config.name,
        tool_filter=tool_filter,
        cache_tools_list=True,
    )


def _mcp_result_to_tool_output(server: MCPServer, result: Any) -> Any:
    """Serialize a ``CallToolResult`` to a tool output, mirroring the agents SDK.

    This reproduces the serialization in ``agents.mcp.util.MCPUtil.invoke_mcp_tool``
    (structured-content JSON when the server asks for it, otherwise text/image
    content blocks, unwrapping a single block). Because the dispatch tool routes
    its own call, this is what makes the agent see byte-identical content to what
    the SDK would have produced building the tool itself.
    """
    if getattr(server, "use_structured_content", False) and result.structuredContent:
        return json.dumps(result.structuredContent)

    outputs: list[dict[str, Any]] = []
    for item in result.content:
        if item.type == "text":
            outputs.append({"type": "text", "text": item.text})
        elif item.type == "image":
            outputs.append(
                {"type": "image", "image_url": f"data:{item.mimeType};base64,{item.data}"}
            )
        else:
            outputs.append({"type": "text", "text": str(item.model_dump(mode="json"))})
    if len(outputs) == 1:
        return outputs[0]
    return outputs


async def dispatch_mcp_call(
    server: MCPServer,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    label: str,
    result_transform: ResultTransform | None = None,
) -> Any:
    """Run one MCP tool call and convert its result to a tool output.

    Shared single dispatch point for the generic ``call_mcp`` tool. Calls
    ``server.call_tool`` with the tool's unprefixed name, then:

    - with a ``result_transform`` (strix-pro's sanitizer), hands the parsed
      :class:`CallToolResult` to it as ``result_transform(label, structured)`` and
      returns whatever the transform returns; or
    - without one, serializes the result the way the agents SDK does (see
      :func:`_mcp_result_to_tool_output`) and, when the result is an MCP error,
      tags the returned output dict with ``success: False`` so the TUI can tell it
      from a success. That tag rides on the human-facing status only: the SDK
      re-projects the value through its ToolOutput schema before the agent sees
      it, which keeps just the known ``type``/``text`` fields and drops
      ``success``, so the agent receives exactly the error content it would have.
    """
    result = await server.call_tool(tool_name, arguments)
    if result_transform is not None:
        return result_transform(label, result.model_dump(mode="json"))
    tool_output = _mcp_result_to_tool_output(server, result)
    if getattr(result, "isError", False) and isinstance(tool_output, dict):
        return {**tool_output, "success": False}
    return tool_output


async def _count_server_tools(config: McpConnectionConfig, server: MCPServer) -> int:
    """Count a connected server's reachable tools for the startup summary.

    ``allowed_tools`` of ``None`` counts every listed tool; a list counts only
    those names. The count matches what ``describe_mcp`` will show, because the
    static tool filter built in :func:`_build_server` restricts the server's own
    ``list_tools`` to the same allowlist.
    """
    allowed = config.allowed_tools
    mcp_tools = await server.list_tools()
    return sum(1 for mcp_tool in mcp_tools if allowed is None or mcp_tool.name in allowed)


async def connect_mcp_servers(
    configs: list[McpConnectionConfig],
) -> list[ConnectedMcpServer]:
    """Connect to each MCP server and return its live session.

    Returns one :class:`ConnectedMcpServer` per server that connected, carrying
    the SDK server (so the caller can clean it up when the run ends and hand it to
    the run's registry) plus the server name, how many tools it offers, and the
    connection's notes. Connections that fail are skipped rather than raised.

    Nothing is registered as an agent tool: the caller builds a per-run
    :class:`~strix.tools.mcp.registry.McpRegistry` from these sessions, and the
    agent reaches each tool on demand through ``describe_mcp`` / ``call_mcp``.
    """
    connected: list[ConnectedMcpServer] = []
    for config in configs:
        server: MCPServer | None = None
        try:
            server = _build_server(config)
            await server.connect()  # type: ignore[no-untyped-call]
            tool_count = await _count_server_tools(config, server)
        except Exception:
            logger.exception("Skipping MCP connection %r", config.name)
            if server is not None:
                with contextlib.suppress(Exception):
                    await server.cleanup()  # type: ignore[no-untyped-call]
            continue
        except BaseException:
            # A cancellation (or other non-Exception failure) mid-connect must not
            # orphan MCP subprocesses or HTTP sessions. Clean up the server being
            # connected and every server already connected, then re-raise so the
            # caller still stops. The runner only receives the list on a clean
            # return, so on an abnormal exit this function owns the cleanup.
            if server is not None:
                with contextlib.suppress(Exception):
                    await server.cleanup()  # type: ignore[no-untyped-call]
            for established in connected:
                with contextlib.suppress(Exception):
                    await established.server.cleanup()  # type: ignore[no-untyped-call]
            raise

        logger.info("Connected MCP server %r (%d tools)", config.name, tool_count)
        connected.append(
            ConnectedMcpServer(
                server=server, name=config.name, tool_count=tool_count, notes=config.notes
            )
        )

    return connected
