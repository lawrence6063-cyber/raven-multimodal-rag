"""get_document_summary tool (E5) — return title/summary/tags for a doc_id.

The summary source is an injectable callable ``doc_id -> dict | None`` so the
tool is unit-testable without a real vector store. A default provider can be
built from Settings that reads enriched chunk metadata from the vector store.
A missing document yields a JSON-RPC invalid-params error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from src.core.response.mcp_types import MCPToolResult, TextContent
from src.mcp_server.protocol_handler import INVALID_PARAMS, JsonRpcError
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = get_logger("mcp_server.tools.get_document_summary")

# TOOL_NAME registered MCP tool name
TOOL_NAME = "get_document_summary"
# TOOL_DESCRIPTION human-readable description for tools/list
TOOL_DESCRIPTION = (
    "Get the title, summary, and tags for a document by its doc_id."
)
# INPUT_SCHEMA JSON Schema for the tool arguments
INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {"type": "string", "description": "The document identifier."},
    },
    "required": ["doc_id"],
}

# SummaryProvider resolves a doc_id to a metadata dict, or None if not found
SummaryProvider = Callable[[str], "dict[str, Any] | None"]


class GetDocumentSummaryTool:
    """Tool object encapsulating the get_document_summary workflow."""

    NAME = TOOL_NAME
    DESCRIPTION = TOOL_DESCRIPTION
    INPUT_SCHEMA = INPUT_SCHEMA

    def __init__(self, summary_provider: SummaryProvider):
        self._provider = summary_provider

    def run(self, doc_id: str) -> MCPToolResult:
        """Return a structured summary for the given doc_id.

        Args:
            doc_id: The document identifier.

        Returns:
            MCPToolResult with Markdown text and structured summary.

        Raises:
            JsonRpcError: -32602 if the document is not found.
        """
        info = self._provider(doc_id)
        if not info:
            raise JsonRpcError(INVALID_PARAMS, f"Document not found: {doc_id}")

        title = str(info.get("title") or "(untitled)")
        summary = str(info.get("summary") or "")
        tags = info.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        created_at = info.get("created_at")

        structured = {
            "doc_id": doc_id,
            "title": title,
            "summary": summary,
            "tags": list(tags),
            "created_at": created_at,
        }

        lines = [f"## {title}", ""]
        if summary:
            lines += [summary, ""]
        if tags:
            lines.append("**Tags:** " + ", ".join(str(t) for t in tags))
        if created_at:
            lines.append(f"**Created:** {created_at}")

        return MCPToolResult(
            content=[TextContent(text="\n".join(lines))],
            structured_content=structured,
            is_error=False,
        )


def build_default_summary_provider(settings: "Settings") -> SummaryProvider:
    """Build a provider that reads doc metadata from the vector store.

    Looks up a single chunk whose metadata ``doc_id`` matches and returns its
    title/summary/tags. Any backend error degrades to ``None`` (not found).
    """
    from src.libs.vector_store.vector_store_factory import VectorStoreFactory

    store = VectorStoreFactory.create(settings.vector_store)

    def _provider(doc_id: str) -> dict[str, Any] | None:
        getter = getattr(store, "get_by_metadata", None)
        if getter is None:
            logger.warning("Vector store has no get_by_metadata; cannot resolve summary")
            return None
        try:
            records = getter({"doc_id": doc_id}, limit=1)
        except Exception as e:
            logger.warning(f"Summary lookup failed for {doc_id}: {e}")
            return None
        if not records:
            return None
        return records[0].metadata or {}

    return _provider
