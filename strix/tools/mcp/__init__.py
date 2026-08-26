"""Generic MCP client: connect MCP servers and reach their tools on demand."""

from __future__ import annotations

from strix.tools.mcp.agent_tools import call_mcp, describe_mcp
from strix.tools.mcp.client import ConnectedMcpServer, connect_mcp_servers
from strix.tools.mcp.config import (
    BearerAuth,
    McpAuth,
    McpConnectionConfig,
)
from strix.tools.mcp.loader import load_user_mcp_configs
from strix.tools.mcp.naming import namespaced_tool_name
from strix.tools.mcp.registry import (
    MCP_REGISTRY_CONTEXT_KEY,
    McpConnectionEntry,
    McpConnectionSummary,
    McpRegistry,
    mcp_inventory_context,
)


__all__ = [
    "MCP_REGISTRY_CONTEXT_KEY",
    "BearerAuth",
    "ConnectedMcpServer",
    "McpAuth",
    "McpConnectionConfig",
    "McpConnectionEntry",
    "McpConnectionSummary",
    "McpRegistry",
    "call_mcp",
    "connect_mcp_servers",
    "describe_mcp",
    "load_user_mcp_configs",
    "mcp_inventory_context",
    "namespaced_tool_name",
]
