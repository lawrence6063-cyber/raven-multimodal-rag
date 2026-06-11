"""LLM Reranker — uses LLM to score and rerank candidates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.libs.reranker.base_reranker import BaseReranker, RerankCandidate, RerankerError
from src.libs.reranker.reranker_factory import register_reranker

if TYPE_CHECKING:
    from src.core.settings import RerankSettings


@register_reranker("llm")
class LLMReranker(BaseReranker):
    """Reranker that uses LLM to score candidate relevance."""

    def __init__(self, settings: "RerankSettings"):
        self._settings = settings
        self._top_n = settings.top_n
        self._prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        """Load rerank prompt template from file."""
        prompt_path = Path("config/prompts/rerank.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        # Fallback prompt
        return (
            "Given a query and candidates, score each candidate's relevance 0-10.\n"
            "Query: {query}\nCandidates:\n{candidates}\n"
            "Output JSON array: [{\"index\": 0, \"score\": 8}, ...]"
        )

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        """Rerank using LLM scoring."""
        if not candidates:
            return []

        from src.libs.llm.llm_factory import LLMFactory
        from src.libs.llm.base_llm import ChatMessage, LLMError
        from src.core.settings import LLMSettings, load_settings

        try:
            settings = load_settings()
            llm = LLMFactory.create(settings.llm)
        except Exception as e:
            raise RerankerError(f"Cannot create LLM for reranking: {e}", provider="llm") from e

        # Build prompt
        candidates_text = "\n".join(
            f"[{i}] {c.text[:300]}" for i, c in enumerate(candidates)
        )
        prompt = self._prompt_template.format(query=query, candidates=candidates_text)

        try:
            response = llm.chat([ChatMessage(role="user", content=prompt)])
            scores = self._parse_scores(response.content, len(candidates))
        except (LLMError, Exception) as e:
            raise RerankerError(f"LLM rerank failed: {e}", provider="llm") from e

        # Apply scores and sort
        for idx, score in scores:
            if 0 <= idx < len(candidates):
                candidates[idx].score = score

        ranked = sorted(candidates, key=lambda c: c.score, reverse=True)
        return ranked[: self._top_n]

    def _parse_scores(self, response_text: str, num_candidates: int) -> list[tuple[int, float]]:
        """Parse LLM response into (index, score) pairs."""
        try:
            # Try to extract JSON array from response
            text = response_text.strip()
            # Find JSON array boundaries
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
                return [(item["index"], float(item["score"])) for item in data]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: return original order with equal scores
        return [(i, 5.0) for i in range(num_candidates)]

    @property
    def provider_name(self) -> str:
        return "llm"
