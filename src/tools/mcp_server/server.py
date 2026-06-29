"""MCP Server: exposes PaperPilot tools via the Model Context Protocol.

This allows external AI agents (Claude, GPT, etc.) to call PaperPilot's
research tools through the standardized MCP interface.

Usage:
    python -m src.tools.mcp_server.server
"""

from __future__ import annotations

import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.tools.arxiv_tool import ArxivSearchTool
from src.tools.github_tool import GitHubSearchTool
from src.tools.jina_reader import JinaReaderTool
from src.tools.web_search_tool import WebSearchTool

app = Server("paperpilot")

# ──────────────────────────────────────────────
# Tool definitions (MCP schema)
# ──────────────────────────────────────────────

TOOLS: list[Tool] = [
    Tool(
        name="arxiv_search",
        description="Search Arxiv for academic papers by topic or keyword",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ),
    Tool(
        name="github_search",
        description="Search GitHub repositories by topic, language, or keyword",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ),
    Tool(
        name="web_search",
        description="Search the web using DuckDuckGo for general information",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    ),
    Tool(
        name="jina_reader",
        description="Read and extract clean text content from a web URL",
        inputSchema={
            "type": "object",
            "properties": {"url": {"type": "string", "description": "Full URL to read"}},
            "required": ["url"],
        },
    ),
]

_TOOL_MAP = {
    "arxiv_search": ArxivSearchTool,
    "github_search": GitHubSearchTool,
    "web_search": WebSearchTool,
    "jina_reader": JinaReaderTool,
}


# ──────────────────────────────────────────────
# MCP Handlers
# ──────────────────────────────────────────────


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    tool_cls = _TOOL_MAP.get(name)
    if not tool_cls:
        return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]

    tool = tool_cls()
    query = arguments.get("query", arguments.get("url", ""))

    try:
        results = tool.run(query)
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False, indent=2))]
    except Exception as exc:
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
