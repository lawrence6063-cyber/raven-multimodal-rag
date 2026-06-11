"""Integration tests for the MCP Server (E1, E3, E6).

In-memory tests drive the real ``Server`` (wired with injected fake backends)
through a connected ClientSession, exercising tools/list, query_knowledge_hub
(citations) and multimodal image returns. A subprocess test spawns the actual
``main.py`` over stdio to verify the initialize handshake and that stdout stays
free of log pollution.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.settings import Settings
from src.core.trace.trace_collector import TraceCollector
from src.core.types import RetrievalResult
from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.server import create_server
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _FakeHybridSearch:
    """Returns canned retrieval results regardless of query."""

    def __init__(self, results: list[RetrievalResult]):
        self._results = results

    def search(self, query="", top_k=None, filters=None, trace=None, image=None):
        return list(self._results)


class _FakeStorage:
    """Maps image ids to on-disk paths for MultimodalAssembler."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def get_path(self, image_id: str):
        return self._mapping.get(image_id)


def _disabled_collector() -> TraceCollector:
    """A TraceCollector with persistence disabled (no file writes in tests)."""
    settings = SimpleNamespace(
        observability=SimpleNamespace(trace_enabled=False, log_file="")
    )
    return TraceCollector(settings)


def _build_handler(query_results, image_storage=None) -> ProtocolHandler:
    handler = ProtocolHandler()
    multimodal = MultimodalAssembler(image_storage) if image_storage else None
    tool = QueryKnowledgeHubTool(
        Settings(),
        hybrid_search=_FakeHybridSearch(query_results),
        multimodal_assembler=multimodal,
        trace_collector=_disabled_collector(),
    )
    handler.register(tool.NAME, tool.DESCRIPTION, tool.INPUT_SCHEMA, tool.run)
    return handler


def _run(scenario):
    """Run an async scenario coroutine function under anyio."""
    anyio.run(scenario)


class TestInMemoryTools:
    def test_tools_list_exposes_query_tool(self):
        handler = _build_handler([])
        server = create_server(handler)

        async def scenario():
            async with create_connected_server_and_client_session(server) as session:
                listed = await session.list_tools()
                names = {t.name for t in listed.tools}
                assert "query_knowledge_hub" in names

        _run(scenario)

    def test_query_knowledge_hub_returns_citations(self):
        results = [
            RetrievalResult(
                chunk_id="docA_0001",
                score=0.9,
                text="Foxes are clever animals.",
                metadata={"source_path": "a.pdf", "page": 2, "doc_id": "docA"},
            )
        ]
        server = create_server(_build_handler(results))

        async def scenario():
            async with create_connected_server_and_client_session(server) as session:
                result = await session.call_tool(
                    "query_knowledge_hub", {"query": "tell me about foxes"}
                )
                assert result.isError is False
                assert result.content[0].type == "text"
                assert "[1]" in result.content[0].text
                citations = result.structuredContent["citations"]
                assert citations[0]["chunk_id"] == "docA_0001"
                assert citations[0]["source"] == "a.pdf"

        _run(scenario)

    def test_query_knowledge_hub_empty_results_friendly(self):
        server = create_server(_build_handler([]))

        async def scenario():
            async with create_connected_server_and_client_session(server) as session:
                result = await session.call_tool("query_knowledge_hub", {"query": "x"})
                assert result.isError is False
                assert "No relevant results" in result.content[0].text

        _run(scenario)

    def test_query_knowledge_hub_returns_image(self, tmp_path):
        png = tmp_path / "fig1.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n fake")
        results = [
            RetrievalResult(
                chunk_id="docB_0001",
                score=0.8,
                text="See figure.",
                metadata={"source_path": "b.pdf", "doc_id": "docB", "image_refs": ["fig1"]},
            )
        ]
        storage = _FakeStorage({"fig1": str(png)})
        server = create_server(_build_handler(results, image_storage=storage))

        async def scenario():
            async with create_connected_server_and_client_session(server) as session:
                result = await session.call_tool("query_knowledge_hub", {"query": "figure"})
                image_items = [c for c in result.content if c.type == "image"]
                assert len(image_items) == 1
                assert image_items[0].mimeType == "image/png"
                assert image_items[0].data

        _run(scenario)


@pytest.mark.integration
class TestSubprocessStdio:
    def test_initialize_and_list_tools_over_stdio(self):
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(_PROJECT_ROOT / "main.py")],
            cwd=str(_PROJECT_ROOT),
        )

        async def scenario():
            async with stdio_client(params) as (read, write):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=30)
                ) as session:
                    init = await session.initialize()
                    assert init.serverInfo.name == "modular-rag-mcp-server"
                    listed = await session.list_tools()
                    names = {t.name for t in listed.tools}
                    assert {
                        "query_knowledge_hub",
                        "list_collections",
                        "get_document_summary",
                    } <= names

        _run(scenario)
