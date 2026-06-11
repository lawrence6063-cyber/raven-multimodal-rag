"""ConfigService — reads and formats Settings for the Dashboard overview (G1)."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from src.core.settings import Settings, load_settings
from src.observability.logger import get_logger

logger = get_logger("dashboard.config_service")


class ConfigService:
    """Provides formatted configuration data for Dashboard display."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings

    @property
    def settings(self) -> Settings:
        """Lazy-load settings on first access."""
        if self._settings is None:
            try:
                self._settings = load_settings()
            except Exception as e:
                logger.warning(f"Failed to load settings: {e}")
                self._settings = Settings()
        return self._settings

    def get_component_cards(self) -> list[dict[str, Any]]:
        """Get summary cards for each configured component.

        Returns:
            List of dicts with component name, provider, and status info.
        """
        s = self.settings
        cards = [
            {
                "name": "LLM",
                "provider": s.llm.provider or "not configured",
                "model": s.llm.model,
                "icon": "🧠",
            },
            {
                "name": "Embedding",
                "provider": s.embedding.provider or "not configured",
                "model": s.embedding.model,
                "icon": "📐",
            },
            {
                "name": "Vector Store",
                "provider": s.vector_store.provider,
                "collection": s.vector_store.collection_name,
                "icon": "🗄️",
            },
            {
                "name": "Splitter",
                "provider": s.splitter.provider,
                "chunk_size": s.splitter.chunk_size,
                "overlap": s.splitter.chunk_overlap,
                "icon": "✂️",
            },
            {
                "name": "Retrieval",
                "top_k": s.retrieval.top_k,
                "rrf_k": s.retrieval.rrf_k,
                "icon": "🔍",
            },
            {
                "name": "Reranker",
                "provider": s.rerank.provider if s.rerank.enabled else "disabled",
                "enabled": s.rerank.enabled,
                "icon": "🔀",
            },
        ]
        return cards

    def get_observability_info(self) -> dict[str, Any]:
        """Get observability configuration info."""
        s = self.settings
        return {
            "trace_enabled": s.observability.trace_enabled,
            "log_file": s.observability.log_file,
            "log_level": s.observability.log_level,
        }

    def get_raw_config(self) -> dict[str, Any]:
        """Get the full configuration as a dict (for detail view)."""
        return asdict(self.settings)
