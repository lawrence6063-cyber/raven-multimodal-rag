"""query_knowledge_hub tool (E3) — main retrieval entry point.

Orchestrates the Core retrieval stack (HybridSearch + optional QueryReranker),
builds a cited Markdown + structuredContent response via ResponseBuilder, and
appends any associated images as Base64 ImageContent via MultimodalAssembler.

Dependencies are injectable for testability; when omitted they are lazily
constructed from Settings so ``server.py`` can wire a ready-to-use instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.response.mcp_types import MCPToolResult
from src.core.response.response_builder import ResponseBuilder
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext
from src.mcp_server.protocol_handler import INVALID_PARAMS, JsonRpcError
from src.mcp_server.tools.image_input import ImageInputError, validate_query_image
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.query_engine.reranker import QueryReranker
    from src.core.response.multimodal_assembler import MultimodalAssembler
    from src.core.settings import Settings

logger = get_logger("mcp_server.tools.query_knowledge_hub")

# TOOL_NAME registered MCP tool name
TOOL_NAME = "query_knowledge_hub"
# TOOL_DESCRIPTION human-readable description for tools/list
TOOL_DESCRIPTION = (
    "Search the knowledge base with hybrid, cross-modal retrieval and reranking. "
    "Provide a text 'query', an 'image' (Base64 or a whitelisted local path), or "
    "both. Returns the most relevant passages with inline citations and any "
    "associated images."
)
# INPUT_SCHEMA JSON Schema for the tool arguments
INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query text."},
        "image": {
            "type": "string",
            "description": (
                "Optional query image as Base64 data (raw or data: URI) or a local "
                "file path under the 'data/' directory. Enables cross-modal search."
            ),
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "minimum": 1,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to restrict the search.",
        },
    },
    # 至少需要 query 或 image 之一，运行时在 run() 中校验。
    "required": [],
}


class QueryKnowledgeHubTool:
    """Tool object encapsulating the query_knowledge_hub workflow."""

    NAME = TOOL_NAME
    DESCRIPTION = TOOL_DESCRIPTION
    INPUT_SCHEMA = INPUT_SCHEMA

    def __init__(
        self,
        settings: "Settings",
        hybrid_search: "HybridSearch | None" = None,
        reranker: "QueryReranker | None" = None,
        response_builder: ResponseBuilder | None = None,
        multimodal_assembler: "MultimodalAssembler | None" = None,
        trace_collector: TraceCollector | None = None,
    ):
        self._settings = settings
        self._hybrid = hybrid_search
        self._reranker = reranker
        self._builder = response_builder or ResponseBuilder()
        self._multimodal = multimodal_assembler
        self._collector = trace_collector or TraceCollector(settings)

    def run(
        self,
        query: str = "",
        top_k: int | None = None,
        collection: str | None = None,
        image: str | None = None,
    ) -> MCPToolResult:
        """Execute retrieval and build the MCP tool result.

        Args:
            query: User search query text (optional when ``image`` is given).
            top_k: Optional result-count override.
            collection: Optional collection filter.
            image: Optional query image (Base64 or a whitelisted local path).

        Returns:
            MCPToolResult with cited Markdown text (and images when present).

        Raises:
            JsonRpcError: -32602 if neither query nor image is provided, or the
                image input is malformed/disallowed.
        """
        query = query or ""
        if not query and not image:
            raise JsonRpcError(
                INVALID_PARAMS, "Provide at least one of 'query' or 'image'"
            )

        image_input: str | bytes | None = None
        if image:
            try:
                image_input = validate_query_image(image)
            except ImageInputError as e:
                raise JsonRpcError(INVALID_PARAMS, f"Invalid image: {e}")

        trace = TraceContext(trace_type="query")
        hybrid = self._get_hybrid()

        filters = {"collection": collection} if collection else None
        results = hybrid.search(
            query=query, top_k=top_k, filters=filters, trace=trace, image=image_input
        )
        logger.info(f"query_knowledge_hub: {len(results)} candidates")

        # Reranking is text-driven; skip it for image-only queries.
        if query and self._settings.rerank.enabled:
            reranker = self._get_reranker()
            if reranker is not None:
                results = reranker.rerank(query, results, trace=trace)

        result = self._builder.build(results, query)

        if self._multimodal is not None and results:
            try:
                images = self._multimodal.assemble(results)
                result.content.extend(images)
            except Exception as e:  # never block the text answer
                logger.warning(f"Multimodal assembly failed, text-only: {e}")

        trace.finish()
        self._collector.collect(trace)

        return result

    def _get_hybrid(self) -> "HybridSearch":
        """Lazily build HybridSearch from settings when not injected."""
        if self._hybrid is None:
            from src.core.query_engine.hybrid_search import HybridSearch

            self._hybrid = HybridSearch(self._settings)
        return self._hybrid

    def _get_reranker(self) -> "QueryReranker | None":
        """Lazily build QueryReranker from settings when not injected."""
        if self._reranker is None:
            try:
                from src.core.query_engine.reranker import QueryReranker

                self._reranker = QueryReranker(self._settings)
            except Exception as e:
                logger.warning(f"Reranker unavailable, skipping: {e}")
                return None
        return self._reranker
