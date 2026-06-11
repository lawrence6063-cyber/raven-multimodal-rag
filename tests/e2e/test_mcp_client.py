"""E2E: MCP client-side invocation over a real subprocess transport (I1).

Spawns the MCP server in a child process and drives it through the official
MCP Python SDK ``ClientSession`` over stdio, exercising the full client flow:
``initialize`` -> ``tools/list`` -> ``tools/call`` for ``query_knowledge_hub``,
asserting that citations are returned.

To keep the test hermetic (no network / no real embeddings / no API keys) the
child process runs a tiny driver that wires the *real* production server stack
(``ProtocolHandler`` + ``QueryKnowledgeHubTool`` + the SDK ``run_stdio`` adapter)
with an injected deterministic fake ``HybridSearch``. Only the retrieval backend
is faked; the protocol handling, tool execution, citation building and stdio
JSON-RPC framing are all the genuine production code paths.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import anyio
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Driver executed in the child process. It builds the real server with a fake,
# deterministic retrieval backend and serves over stdio. Kept as a standalone
# module file (written to tmp) so the subprocess imports `src.*` from the
# project root on its sys.path.
_DRIVER_BODY = '''\
"""Hermetic MCP server driver for the I1 E2E test (fake retrieval backend)."""

from types import SimpleNamespace

import anyio

from src.core.settings import Settings
from src.core.trace.trace_collector import TraceCollector
from src.core.types import RetrievalResult
from src.mcp_server.protocol_handler import ProtocolHandler
from src.mcp_server.server import run_stdio
from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool


class _FakeHybridSearch:
    """Returns canned retrieval results regardless of the query text."""

    def search(self, query="", top_k=None, filters=None, trace=None, image=None):
        return [
            RetrievalResult(
                chunk_id="docFox_0001",
                score=0.95,
                text="Foxes are clever, agile animals found across many regions.",
                metadata={"source_path": "wildlife.pdf", "page": 7, "doc_id": "docFox"},
            ),
            RetrievalResult(
                chunk_id="docFox_0002",
                score=0.81,
                text="A fox uses its bushy tail for balance and warmth.",
                metadata={"source_path": "wildlife.pdf", "page": 8, "doc_id": "docFox"},
            ),
        ]


def _disabled_collector() -> TraceCollector:
    settings = SimpleNamespace(
        observability=SimpleNamespace(trace_enabled=False, log_file="")
    )
    return TraceCollector(settings)


def _build_handler() -> ProtocolHandler:
    handler = ProtocolHandler()
    tool = QueryKnowledgeHubTool(
        Settings(),
        hybrid_search=_FakeHybridSearch(),
        trace_collector=_disabled_collector(),
    )
    handler.register(tool.NAME, tool.DESCRIPTION, tool.INPUT_SCHEMA, tool.run)
    return handler


if __name__ == "__main__":
    anyio.run(run_stdio, _build_handler())
'''


def _write_driver(tmp_path: Path) -> Path:
    """Write the hermetic server driver script and return its path.

    A small bootstrap header makes the project root importable, since a script
    launched from ``tmp_path`` has its own directory (not the cwd) on sys.path.
    """
    bootstrap = (
        "import sys\n"
        f"sys.path.insert(0, {str(_PROJECT_ROOT)!r})\n\n"
    )
    driver = tmp_path / "mcp_server_driver.py"
    driver.write_text(bootstrap + _DRIVER_BODY, encoding="utf-8")
    return driver


@pytest.mark.e2e
class TestMcpClientSubprocess:
    """Drive the MCP server over a real subprocess stdio transport."""

    def test_query_knowledge_hub_returns_citations(self, tmp_path):
        driver = _write_driver(tmp_path)
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(driver)],
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
                    assert "query_knowledge_hub" in names

                    result = await session.call_tool(
                        "query_knowledge_hub", {"query": "tell me about foxes"}
                    )
                    assert result.isError is False

                    # Cited Markdown text body.
                    text_items = [c for c in result.content if c.type == "text"]
                    assert text_items, "expected a text content block"
                    assert "[1]" in text_items[0].text

                    # Structured citations.
                    citations = result.structuredContent["citations"]
                    assert len(citations) >= 1
                    assert citations[0]["chunk_id"] == "docFox_0001"
                    assert citations[0]["source"] == "wildlife.pdf"

        anyio.run(scenario)

    def test_query_with_top_k_argument(self, tmp_path):
        driver = _write_driver(tmp_path)
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(driver)],
            cwd=str(_PROJECT_ROOT),
        )

        async def scenario():
            async with stdio_client(params) as (read, write):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=30)
                ) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        "query_knowledge_hub",
                        {"query": "foxes", "top_k": 2},
                    )
                    assert result.isError is False
                    citations = result.structuredContent["citations"]
                    assert len(citations) == 2

        anyio.run(scenario)
