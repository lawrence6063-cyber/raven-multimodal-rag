"""QueryProcessor — preprocesses user queries for retrieval.

Extracts keywords, parses filters, and prepares the query for
both dense and sparse retrieval paths.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProcessedQuery:
    """Result of query preprocessing."""

    original: str
    keywords: list[str]
    filters: dict[str, Any] = field(default_factory=dict)


class QueryProcessor:
    """Processes raw user queries into structured retrieval inputs."""

    # Common stop words to filter out of keyword extraction
    STOP_WORDS = frozenset([
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "to", "of", "in", "for", "on",
        "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off",
        "and", "but", "or", "if", "this", "that", "these", "those", "it",
        "what", "which", "who", "whom", "how", "when", "where", "why",
        "not", "no", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "very", "so", "than", "too",
        "的", "是", "了", "在", "有", "和", "就", "不", "人", "都",
        "一", "我", "他", "这", "中", "大", "来", "上", "个", "为",
    ])

    def process(self, query: str, filters: dict[str, Any] | None = None) -> ProcessedQuery:
        """Process a raw query string.

        Args:
            query: User's raw query text.
            filters: Optional pre-defined filters (e.g., collection).

        Returns:
            ProcessedQuery with extracted keywords and filters.
        """
        keywords = self._extract_keywords(query)
        parsed_filters = filters or {}

        # Extract inline filter patterns like "collection:xxx"
        inline_filters = self._parse_inline_filters(query)
        parsed_filters.update(inline_filters)

        return ProcessedQuery(
            original=query,
            keywords=keywords,
            filters=parsed_filters,
        )

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract meaningful keywords from query text."""
        # Tokenize: split on whitespace and punctuation
        tokens = re.findall(r'[\w\u4e00-\u9fff]+', text.lower())
        # Filter stop words and short tokens
        keywords = [t for t in tokens if t not in self.STOP_WORDS and len(t) > 1]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)
        return unique

    def _parse_inline_filters(self, query: str) -> dict[str, Any]:
        """Parse inline filter syntax like 'collection:finance'."""
        filters = {}
        pattern = r'\b(collection|source|doc_type):(\S+)'
        for match in re.finditer(pattern, query, re.IGNORECASE):
            filters[match.group(1).lower()] = match.group(2)
        return filters
