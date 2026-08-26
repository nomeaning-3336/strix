"""The two generic MCP dispatch tools every agent carries.

Under the generic-dispatch model an agent does not get one tool per MCP tool.
It gets exactly these two, plus a short inventory in its system prompt naming
which connections exist and what each is for (no schemas):

- ``describe_mcp(connection)`` returns, as text, one connection's tools with
  their names, descriptions, and JSON input schemas — the schemas the model
  needs, fetched on demand instead of loaded onto every request up front.
- ``call_mcp(connection, tool, arguments)`` dispatches one call to a
  connection's tool and returns its result.

Both read the per-run :class:`~strix.tools.mcp.registry.McpRegistry` from the run
context under :data:`~strix.tools.mcp.registry.MCP_REGISTRY_CONTEXT_KEY`. They are
ordinary ``FunctionTool`` objects placed in the agent factory's base tool set, so
the factory's output-bounding and disk-spill wrapping apply to their results
automatically.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agents import RunContextWrapper, function_tool

from strix.tools.mcp.client import dispatch_mcp_call
from strix.tools.mcp.naming import namespaced_tool_name
from strix.tools.mcp.registry import MCP_REGISTRY_CONTEXT_KEY, McpRegistry


if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool


def _registry_from_ctx(ctx: RunContextWrapper) -> McpRegistry | None:
    context = ctx.context if isinstance(ctx.context, dict) else {}
    registry = context.get(MCP_REGISTRY_CONTEXT_KEY)
    return registry if isinstance(registry, McpRegistry) else None


_NO_CONNECTIONS = "No MCP connections are configured for this run."


def _unknown_connection(connection: str, registry: McpRegistry) -> str:
    available = ", ".join(registry.names()) or "(none)"
    return f"Unknown MCP connection {connection!r}. Available connections: {available}."


def _format_tool(tool: MCPTool) -> str:
    schema = json.dumps(tool.inputSchema or {"type": "object"}, indent=2, ensure_ascii=False)
    description = (tool.description or "").strip() or "(no description)"
    return f"- {tool.name}: {description}\n  input schema:\n{schema}"


@function_tool(timeout=60)
async def describe_mcp(ctx: RunContextWrapper, connection: str) -> str:
    """List the tools one MCP connection offers, with their input schemas.

    Read-only. Look up a connection by the name shown in the MCP inventory in
    your system prompt; this returns each of its tools with the tool's name,
    description, and JSON input schema — the argument shape you pass to
    ``call_mcp``. Call this before ``call_mcp`` on any connection you have not
    used yet. Nothing is fetched from or run against the connection's data.

    Args:
        connection: The connection name exactly as shown in the MCP inventory.
    """
    registry = _registry_from_ctx(ctx)
    if registry is None or not registry:
        return _NO_CONNECTIONS
    entry = registry.get(connection)
    if entry is None:
        return _unknown_connection(connection, registry)
    tools = await entry.server.list_tools()
    if not tools:
        return f"MCP connection {connection!r} offers no tools."
    header = f"MCP connection {connection!r} offers {len(tools)} tool(s):"
    body = "\n".join(_format_tool(tool) for tool in tools)
    return f"{header}\n{body}"


@function_tool(timeout=120, strict_mode=False)
async def call_mcp(
    ctx: RunContextWrapper,
    connection: str,
    tool: str,
    arguments: Any = None,
) -> Any:
    """Call one tool on one MCP connection and return its result.

    Address the tool by the connection name from the MCP inventory and the tool
    name from ``describe_mcp`` on that connection. Pass the tool's arguments as an
    object matching the input schema ``describe_mcp`` showed for it (omit it, or
    pass an empty object, for a tool that takes no arguments).

    Args:
        connection: The connection name exactly as shown in the MCP inventory.
        tool: The tool name, exactly as reported by ``describe_mcp``.
        arguments: The tool's arguments as an object of names to values, or
            omitted/empty for a tool that takes none. ``arguments`` is passed
            through as-is, so its shape is whatever ``describe_mcp`` showed for
            the tool rather than a shape this tool fixes in advance.
    """
    registry = _registry_from_ctx(ctx)
    if registry is None or not registry:
        return _NO_CONNECTIONS
    entry = registry.get(connection)
    if entry is None:
        return _unknown_connection(connection, registry)
    if arguments is not None and not isinstance(arguments, dict):
        return (
            f"Invalid arguments for {connection!r}.{tool}: expected a JSON object of "
            "argument names to values, or none. Call describe_mcp for the input schema."
        )
    available = await entry.server.list_tools()
    valid_names = {mcp_tool.name for mcp_tool in available}
    if tool not in valid_names:
        offered = ", ".join(sorted(valid_names)) or "(none)"
        return (
            f"Unknown tool {tool!r} on MCP connection {connection!r}. "
            f"Tools this connection offers: {offered}. "
            "Call describe_mcp for their input schemas."
        )
    return await dispatch_mcp_call(
        entry.server,
        tool,
        arguments or {},
        label=namespaced_tool_name(connection, tool),
        result_transform=entry.result_transform,
    )
