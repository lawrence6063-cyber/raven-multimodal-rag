"""Recursive Splitter — wraps LangChain RecursiveCharacterTextSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.splitter.base_splitter import BaseSplitter, SplitterError
from src.libs.splitter.splitter_factory import register_splitter

if TYPE_CHECKING:
    from src.core.settings import SplitterSettings


@register_splitter("recursive")
class RecursiveSplitter(BaseSplitter):
    """Recursive character text splitter optimized for Markdown documents."""

    # Markdown-aware separators (ordered by priority)
    MARKDOWN_SEPARATORS = [
        "\n## ",      # H2 headers
        "\n### ",     # H3 headers
        "\n#### ",    # H4 headers
        "\n\n",       # Paragraphs
        "\n",         # Lines
        ". ",         # Sentences
        " ",          # Words
        "",           # Characters (last resort)
    ]

    def __init__(self, settings: "SplitterSettings"):
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text using recursive character splitting with Markdown separators."""
        if not text or not text.strip():
            return []

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                separators=self.MARKDOWN_SEPARATORS,
                keep_separator=True,
                strip_whitespace=True,
            )
            chunks = splitter.split_text(text)
            return [c for c in chunks if c.strip()]

        except ImportError:
            raise SplitterError(
                "langchain-text-splitters not installed. Run: pip install langchain-text-splitters",
                provider="recursive",
            )
        except Exception as e:
            raise SplitterError(f"Split failed: {e}", provider="recursive") from e

    @property
    def provider_name(self) -> str:
        return "recursive"
