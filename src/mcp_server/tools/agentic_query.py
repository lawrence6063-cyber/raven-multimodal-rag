"""agentic_query tool — Agentic RAG entry point (OPTIMIZATION_SPEC §3.5).

Unlike ``query_knowledge_hub`` (which returns retrieved passages for an external
client LLM to synthesize), this tool runs a server-side agent that routes,
retrieves (multi-hop), self-corrects, and returns a *synthesized answer* with
inline ``[n]`` citations plus supporting images.

When ``settings.agent.enabled`` is False the tool delegates to
``query_knowledge_hub`` so the tool list stays stable and behavior is unchanged.

Dependencies are injectable for testability; when omitted they are lazily built
from Settings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.response.mcp_types import MCPToolResult, TextContent
from src.core.response.response_builder import ResponseBuilder
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext
from src.mcp_server.protocol_handler import INVALID_PARAMS, JsonRpcError
from src.mcp_server.tools.image_input import ImageInputError, validate_query_image
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.agent.agentic_rag import AgenticRAG
    from src.core.response.multimodal_assembler import MultimodalAssembler
    from src.core.settings import Settings
    from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

logger = get_logger("mcp_server.tools.agentic_query")

# TOOL_NAME registered MCP tool name
TOOL_NAME = "agentic_query"
# TOOL_DESCRIPTION human-readable description for tools/list
TOOL_DESCRIPTION = (
    "Answer a question with an autonomous retrieval agent: it decides whether and "
    "where to search, decomposes complex questions, retrieves over multiple hops, "
    "self-corrects when evidence is insufficient, and returns a synthesized answer "
    "with inline [n] citations and supporting images. Use for multi-hop or "
    "reasoning-heavy questions; use 'query_knowledge_hub' for raw passage lookup."
)
# INPUT_SCHEMA JSON Schema for the tool arguments
INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "User question (natural language)."},
        "collection": {
            "type": "string",
            "description": "Optional collection name to restrict the search.",
        },
        "image": {
            "type": "string",
            "description": (
                "Optional query image as Base64 data (raw or data: URI) or a local "
                "file path under 'data/'. Enables cross-modal retrieval."
            ),
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum results per sub-query.",
            "minimum": 1,
        },
    },
    "required": ["query"],
}


class AgenticQueryTool:
    """Tool object encapsulating the agentic_query workflow."""

    NAME = TOOL_NAME
    DESCRIPTION = TOOL_DESCRIPTION
    INPUT_SCHEMA = INPUT_SCHEMA

    def __init__(
        self,
        settings: "Settings",
        agentic_rag: "AgenticRAG | None" = None,
        response_builder: ResponseBuilder | None = None,
        multimodal_assembler: "MultimodalAssembler | None" = None,
        trace_collector: TraceCollector | None = None,
        delegate: "QueryKnowledgeHubTool | None" = None,
    ):
        self._settings = settings
        self._agent = agentic_rag
        self._builder = response_builder or ResponseBuilder()
        self._multimodal = multimodal_assembler
        self._collector = trace_collector or TraceCollector(settings)
        self._delegate = delegate

    def run(
        self,
        query: str = "",
        top_k: int | None = None,
        collection: str | None = None,
        image: str | None = None,
    ) -> MCPToolResult:
        """Execute the agentic pipeline and build the MCP tool result.

        Args:
            query: User question text.
            top_k: Optional per-query result-count override.
            collection: Optional collection filter.
            image: Optional query image (Base64 or a whitelisted local path).

        Returns:
            MCPToolResult with a synthesized, cited answer (and images when
            present). Falls back to ``query_knowledge_hub`` behavior when the
            agent is disabled.

        Raises:
            JsonRpcError: -32602 when neither query nor image is provided, or the
                image input is malformed/disallowed.
        """
        query = query or ""
        if not query and not image:
            raise JsonRpcError(INVALID_PARAMS, "Provide at least one of 'query' or 'image'")

        # Disabled → delegate to the classic retrieval tool (stable tool list).
        if not self._settings.agent.enabled:
            return self._get_delegate().run(
                query=query, top_k=top_k, collection=collection, image=image
            )

        image_input: str | bytes | None = None
        if image:
            try:
                image_input = validate_query_image(image)
            except ImageInputError as e:
                raise JsonRpcError(INVALID_PARAMS, f"Invalid image: {e}")

        trace = TraceContext(trace_type="query")
        agent_result = self._get_agent().run(
            query=query,
            collection=collection,
            image=image_input,
            top_k=top_k,
            trace=trace,
        )

        result = self._build_result(query, agent_result)

        trace.finish()
        self._collector.collect(trace)
        return result

    def _build_result(self, query: str, agent_result) -> MCPToolResult:
        """Assemble the MCP result: synthesized answer + citations + images."""
        built = self._builder.build(agent_result.results, query)
        structured = built.structured_content or {"query": query, "citations": []}
        citations = structured.get("citations", [])

        answer = agent_result.answer
        if answer:
            parts = [answer]
            references = self._render_references(citations)
            if references:
                parts.append(references)
            built.content[0] = TextContent(text="\n\n".join(parts))

        structured["answer"] = answer
        structured["fallback"] = agent_result.fallback
        built.structured_content = structured

        if self._multimodal is not None and agent_result.results:
            try:
                images = self._multimodal.assemble(agent_result.results)
                built.content.extend(images)
            except Exception as e:  # never block the text answer
                logger.warning(f"Multimodal assembly failed, text-only: {e}")

        return built

    @staticmethod
    def _render_references(citations: list[dict[str, Any]]) -> str:
        """Render a compact Markdown references block from citation dicts."""
        if not citations:
            return ""
        lines = ["### References"]
        for c in citations:
            page = c.get("page")
            page_suffix = f" (page {page})" if page is not None else ""
            score = c.get("score", 0.0)
            lines.append(
                f"[{c.get('id')}] {c.get('source', '')}{page_suffix} — score {score:.4f}"
            )
        return "\n".join(lines)

    def _get_agent(self) -> "AgenticRAG":
        """Lazily build the AgenticRAG orchestrator from settings."""
        if self._agent is None:
            from src.core.agent.agentic_rag import AgenticRAG

            self._agent = AgenticRAG(self._settings)
        return self._agent

    def _get_delegate(self) -> "QueryKnowledgeHubTool":
        """Lazily build the classic query tool used when the agent is disabled."""
        if self._delegate is None:
            from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

            self._delegate = QueryKnowledgeHubTool(
                self._settings, multimodal_assembler=self._multimodal
            )
        return self._delegate
