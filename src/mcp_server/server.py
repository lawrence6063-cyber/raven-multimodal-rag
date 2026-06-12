"""MCP Server entry point (E1) — Stdio Transport.

Wires the SDK-independent ``ProtocolHandler`` into the official MCP Python SDK's
low-level ``Server`` over Stdio. Per DEV_SPEC 3.2.2, ``stdout`` carries only MCP
protocol messages while all logging goes to ``stderr`` (see ``get_logger``).

The SDK handles the JSON-RPC framing, ``initialize`` capability negotiation, and
input-schema validation; this module only adapts tool listing/calling to the
project's tool registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from src.core.settings import SettingsError, load_settings
from src.mcp_server.protocol_handler import (
    SERVER_NAME,
    JsonRpcError,
    ProtocolHandler,
)
from src.mcp_server.tools.agentic_query import AgenticQueryTool
from src.mcp_server.tools.get_document_summary import (
    GetDocumentSummaryTool,
    build_default_summary_provider,
)
from src.mcp_server.tools.list_collections import ListCollectionsTool
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = get_logger("mcp_server.server")


def build_protocol_handler(
    settings: "Settings",
    documents_dir: str = "data/documents",
) -> ProtocolHandler:
    """Construct and register the production tool set.

    Args:
        settings: Loaded application settings.
        documents_dir: Root directory scanned by list_collections.

    Returns:
        A ProtocolHandler with the four core tools registered.
    """
    from src.core.response.multimodal_assembler import MultimodalAssembler
    from src.ingestion.storage.image_storage import ImageStorage

    handler = ProtocolHandler()

    multimodal = MultimodalAssembler(ImageStorage())
    query_tool = QueryKnowledgeHubTool(settings, multimodal_assembler=multimodal)
    handler.register(
        query_tool.NAME, query_tool.DESCRIPTION, query_tool.INPUT_SCHEMA, query_tool.run
    )

    agentic_tool = AgenticQueryTool(settings, multimodal_assembler=multimodal)
    handler.register(
        agentic_tool.NAME,
        agentic_tool.DESCRIPTION,
        agentic_tool.INPUT_SCHEMA,
        agentic_tool.run,
    )

    list_tool = ListCollectionsTool(documents_dir)
    handler.register(
        list_tool.NAME, list_tool.DESCRIPTION, list_tool.INPUT_SCHEMA, list_tool.run
    )

    summary_tool = GetDocumentSummaryTool(build_default_summary_provider(settings))
    handler.register(
        summary_tool.NAME,
        summary_tool.DESCRIPTION,
        summary_tool.INPUT_SCHEMA,
        summary_tool.run,
    )

    return handler


def _to_call_tool_result(payload: dict[str, Any]) -> types.CallToolResult:
    """Convert an MCPToolResult dict into an SDK CallToolResult."""
    content: list[types.ContentBlock] = []
    for item in payload.get("content", []):
        if item.get("type") == "image":
            content.append(
                types.ImageContent(
                    type="image",
                    data=item["data"],
                    mimeType=item.get("mimeType", "image/png"),
                )
            )
        else:
            content.append(types.TextContent(type="text", text=item.get("text", "")))

    return types.CallToolResult(
        content=content,
        structuredContent=payload.get("structuredContent"),
        isError=bool(payload.get("isError", False)),
    )


def create_server(handler: ProtocolHandler) -> Server:
    """Create an SDK Server delegating tools to the given ProtocolHandler.

    Args:
        handler: The tool registry/router.

    Returns:
        A configured (but not yet running) MCP Server.
    """
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in handler.handle_tools_list()["tools"]
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
        try:
            payload = handler.handle_tools_call(name, arguments)
        except JsonRpcError as e:
            logger.warning(f"Tool '{name}' error ({e.code}): {e.message}")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=e.message)],
                isError=True,
            )
        return _to_call_tool_result(payload)

    return server


async def run_stdio(handler: ProtocolHandler) -> None:
    """Run the MCP server over Stdio until the client disconnects."""
    server = create_server(handler)
    init_options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        logger.info(f"MCP server ready over stdio; tools={handler.tool_names}")
        await server.run(read_stream, write_stream, init_options)


def main() -> int:
    """Synchronous entry point: load config, build tools, serve over stdio."""
    try:
        settings = load_settings()
    except SettingsError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    logger.info(
        f"Starting Modular RAG MCP Server "
        f"(LLM={settings.llm.provider}/{settings.llm.model})"
    )
    handler = build_protocol_handler(settings)

    try:
        anyio.run(run_stdio, handler)
    except KeyboardInterrupt:  # pragma: no cover
        logger.info("MCP server interrupted, shutting down")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
