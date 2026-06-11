"""ChunkRefiner — rule-based noise removal + optional LLM enhancement."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from src.core.settings import Settings


class ChunkRefiner(BaseTransform):
    """Refines chunks by removing noise and optionally using LLM for enhancement."""

    def __init__(self, settings: "Settings"):
        self._use_llm = settings.ingestion.chunk_refiner.use_llm
        self._settings = settings
        self._llm = None  # lazily created once, then reused across all chunks
        self._prompt_template = self._load_prompt()

    def _get_llm(self):
        """Lazily create and cache the LLM client (avoid rebuilding per chunk)."""
        if self._llm is None:
            from src.libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(self._settings.llm)
        return self._llm

    def transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Apply rule-based and optional LLM refinement to chunks."""
        refined = []
        for chunk in chunks:
            try:
                new_text = self._rule_based_refine(chunk.text)
                method = "rule"

                if self._use_llm and new_text.strip():
                    llm_result = self._llm_refine(new_text)
                    if llm_result:
                        new_text = llm_result
                        method = "llm"

                new_chunk = Chunk(
                    id=chunk.id,
                    text=new_text,
                    metadata={**chunk.metadata, "refined_by": method},
                    source_ref=chunk.source_ref,
                )
                refined.append(new_chunk)
            except Exception:
                # Preserve original on failure
                chunk.metadata["refined_by"] = "none"
                refined.append(chunk)

        return refined

    def _rule_based_refine(self, text: str) -> str:
        """Apply rule-based cleaning."""
        # Remove excessive whitespace (keep single newlines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove common page headers/footers patterns
        text = re.sub(r'^(Page \d+|第\s*\d+\s*页)[\s]*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-=]{3,}\s*$', '', text, flags=re.MULTILINE)
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        # Collapse multiple spaces (not newlines)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def _llm_refine(self, text: str) -> str | None:
        """Optional LLM-based refinement. Returns None on failure."""
        try:
            from src.libs.llm.base_llm import ChatMessage

            llm = self._get_llm()
            prompt = self._prompt_template.replace("{text}", text)
            response = llm.chat([ChatMessage(role="user", content=prompt)])
            return response.content.strip() if response.content else None
        except Exception:
            return None

    def _load_prompt(self) -> str:
        """Load refinement prompt template."""
        prompt_path = Path("config/prompts/chunk_refinement.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8")
        return "Clean up this text, remove noise:\n{text}"

    @property
    def name(self) -> str:
        return "chunk_refiner"
