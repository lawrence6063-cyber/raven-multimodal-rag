"""CitationGenerator — builds structured citations from retrieval results.

Maps each RetrievalResult into a Citation with a stable 1-based index, pulling
locator fields (source/page/chunk_id/score) from the result metadata. Used by
ResponseBuilder to populate ``structuredContent.citations``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.response.mcp_types import Citation

if TYPE_CHECKING:
    from src.core.types import RetrievalResult

# _SOURCE_KEYS metadata keys probed (in order) to resolve a human-readable source
_SOURCE_KEYS = ("source_path", "file_name", "source", "doc_id")
# _PAGE_KEYS metadata keys probed (in order) to resolve a page number
_PAGE_KEYS = ("page", "page_num", "page_number")
# _TEXT_PREVIEW_LEN maximum number of characters kept in a citation text preview
_TEXT_PREVIEW_LEN = 300


class CitationGenerator:
    """Generates a list of Citation objects from retrieval results."""

    def generate(self, results: list["RetrievalResult"]) -> list[Citation]:
        """Build citations with 1-based ids preserving result order.

        Args:
            results: Ranked retrieval results.

        Returns:
            List of Citation, one per result; empty if results is empty.
        """
        citations: list[Citation] = []
        for index, result in enumerate(results, start=1):
            metadata = result.metadata or {}
            citations.append(
                Citation(
                    id=index,
                    source=self._resolve_source(metadata),
                    chunk_id=result.chunk_id,
                    score=round(float(result.score), 4),
                    page=self._resolve_page(metadata),
                    text=self._preview(result.text),
                )
            )
        return citations

    @staticmethod
    def _resolve_source(metadata: dict) -> str:
        """Return the first available source-like field, or 'unknown'."""
        for key in _SOURCE_KEYS:
            value = metadata.get(key)
            if value:
                return str(value)
        return "unknown"

    @staticmethod
    def _resolve_page(metadata: dict) -> int | None:
        """Return a page number if present and integer-convertible."""
        for key in _PAGE_KEYS:
            value = metadata.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _preview(text: str) -> str:
        """Collapse whitespace and truncate text for compact citations."""
        if not text:
            return ""
        normalized = " ".join(text.split())
        if len(normalized) <= _TEXT_PREVIEW_LEN:
            return normalized
        return normalized[:_TEXT_PREVIEW_LEN].rstrip() + "..."
