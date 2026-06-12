"""Context formatting helpers shared by agent LLM components.

Both the answer synthesizer and the reflector present accumulated retrieval
results to the LLM as a numbered list of truncated passages. Centralising the
format keeps citation indices consistent across components.
"""

from __future__ import annotations

from src.core.types import RetrievalResult

# _DEFAULT_SNIPPET_CHARS per-passage truncation length (mirrors reranker's text[:300])
_DEFAULT_SNIPPET_CHARS = 300


def format_numbered_context(
    context: list[RetrievalResult],
    max_chars: int = _DEFAULT_SNIPPET_CHARS,
) -> str:
    """Render retrieval results as a 1-based numbered passage list.

    Args:
        context: Accumulated retrieval results.
        max_chars: Per-passage truncation length.

    Returns:
        A newline-joined string like ``[1] passage text...``. Empty string when
        there is no context.
    """
    lines = []
    for i, result in enumerate(context, start=1):
        snippet = (result.text or "").strip()[:max_chars]
        lines.append(f"[{i}] {snippet}")
    return "\n".join(lines)
