"""MetadataEnricher — adds title/summary/tags to chunk metadata."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from src.core.settings import Settings


class MetadataEnricher(BaseTransform):
    """Enriches chunk metadata with title, summary, and tags."""

    def __init__(self, settings: "Settings"):
        self._use_llm = settings.ingestion.metadata_enricher.use_llm
        self._settings = settings
        self._llm = None  # lazily created once, then reused across all chunks

    def _get_llm(self):
        """Lazily create and cache the LLM client (avoid rebuilding per chunk)."""
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(self._settings.llm)
        return self._llm

    def transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich chunks with semantic metadata."""
        enriched = []
        for chunk in chunks:
            try:
                meta = dict(chunk.metadata)

                if self._use_llm:
                    llm_meta = self._llm_enrich(chunk.text)
                    if llm_meta:
                        meta.update(llm_meta)
                        meta["enriched_by"] = "llm"
                    else:
                        meta.update(self._rule_enrich(chunk.text))
                        meta["enriched_by"] = "rule"
                else:
                    meta.update(self._rule_enrich(chunk.text))
                    meta["enriched_by"] = "rule"

                enriched.append(Chunk(id=chunk.id, text=chunk.text, metadata=meta, source_ref=chunk.source_ref))
            except Exception:
                chunk.metadata["enriched_by"] = "none"
                enriched.append(chunk)

        return enriched

    def _rule_enrich(self, text: str) -> dict:
        """Rule-based metadata extraction."""
        lines = text.strip().split('\n')
        # Title: first non-empty line (strip markdown headers)
        title = ""
        for line in lines:
            stripped = line.strip().lstrip('#').strip()
            if stripped:
                title = stripped[:100]
                break

        # Summary: first 200 chars
        summary = text[:200].replace('\n', ' ').strip()

        # Tags: extract words that look like keywords (capitalized, technical terms)
        words = text.split()
        tags = list(set(w.strip('.,;:()[]') for w in words if len(w) > 3 and w[0].isupper()))[:5]

        return {"title": title, "summary": summary, "tags": tags}

    def _llm_enrich(self, text: str) -> dict | None:
        """LLM-based metadata extraction. Returns None on failure."""
        try:
            import json
            from src.libs.llm.base_llm import ChatMessage

            llm = self._get_llm()
            prompt = (
                "Extract metadata from this text. Return JSON with keys: title, summary, tags (list).\n"
                f"Text: {text[:500]}\n\nJSON:"
            )
            response = llm.chat([ChatMessage(role="user", content=prompt)])
            data = json.loads(response.content)
            return {"title": data.get("title", ""), "summary": data.get("summary", ""), "tags": data.get("tags", [])}
        except Exception:
            return None

    @property
    def name(self) -> str:
        return "metadata_enricher"
