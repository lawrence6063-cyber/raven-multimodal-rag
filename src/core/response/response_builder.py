"""ResponseBuilder — assembles an MCPToolResult from retrieval results.

Produces the dual-channel response described in DEV_SPEC 3.2.5:
- content[0]: human-readable Markdown with inline ``[n]`` citation markers
  (ensures lowest-common-denominator clients still see references).
- structuredContent.citations: machine-readable citation list for advanced
  clients (source/page/chunk_id/score/text).

Empty results yield a friendly hint instead of an empty payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.response.citation_generator import CitationGenerator
from src.core.response.mcp_types import MCPToolResult, TextContent

if TYPE_CHECKING:
    from src.core.types import RetrievalResult

# _EMPTY_HINT message returned when a query yields no retrieval results
_EMPTY_HINT = (
    "No relevant results were found for your query.\n\n"
    "Hints:\n"
    "- Try rephrasing with different keywords.\n"
    "- Confirm documents have been ingested "
    "(e.g. `python scripts/ingest.py --path <folder>`)."
)


class ResponseBuilder:
    """Builds MCP tool responses with Markdown text and structured citations."""

    def __init__(self, citation_generator: CitationGenerator | None = None):
        self._citations = citation_generator or CitationGenerator()

    def build(self, results: list["RetrievalResult"], query: str) -> MCPToolResult:
        """Assemble an MCPToolResult for the given results and query.

        Args:
            results: Ranked retrieval results (already reranked if enabled).
            query: The original user query, echoed into the Markdown header.

        Returns:
            MCPToolResult with a Markdown text item and citations structured
            content. When results is empty, returns a friendly hint instead.
        """
        if not results:
            return MCPToolResult(
                content=[TextContent(text=_EMPTY_HINT)],
                structured_content={"query": query, "citations": []},
                is_error=False,
            )

        citations = self._citations.generate(results)
        markdown = self._render_markdown(query, citations)

        return MCPToolResult(
            content=[TextContent(text=markdown)],
            structured_content={
                "query": query,
                "citations": [c.to_dict() for c in citations],
            },
            is_error=False,
        )

    @staticmethod
    def _render_markdown(query: str, citations: list) -> str:
        """Render a Markdown answer with a header and a numbered references list."""
        lines = [f'## Results for: "{query}"', ""]
        for citation in citations:
            page_suffix = f", p.{citation.page}" if citation.page is not None else ""
            lines.append(f"**[{citation.id}]** {citation.source}{page_suffix}")
            if citation.text:
                lines.append(f"> {citation.text}")
            lines.append("")

        lines.append("### References")
        for citation in citations:
            page_suffix = f" (page {citation.page})" if citation.page is not None else ""
            lines.append(
                f"[{citation.id}] {citation.source}{page_suffix} "
                f"— score {citation.score:.4f}"
            )

        return "\n".join(lines)
