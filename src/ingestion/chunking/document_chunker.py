"""DocumentChunker — adapts libs.splitter for Document-to-Chunks conversion.

Responsibilities beyond raw text splitting:
1. Chunk ID generation (deterministic)
2. Metadata inheritance from Document
3. chunk_index assignment
4. source_ref linkage
5. Image reference distribution per chunk
6. Type conversion (str -> Chunk objects)
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from src.core.types import Document, Chunk
from src.libs.splitter.splitter_factory import SplitterFactory

if TYPE_CHECKING:
    from src.core.settings import Settings


class DocumentChunker:
    """Converts a Document into a list of Chunks using the configured splitter."""

    def __init__(self, settings: "Settings"):
        self._splitter = SplitterFactory.create(settings.splitter)

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a Document into Chunks with full metadata.

        Args:
            document: The Document to split.

        Returns:
            List of Chunk objects with IDs, metadata, and source references.
        """
        raw_chunks = self._splitter.split_text(document.text)
        if not raw_chunks:
            return []

        chunks = []
        doc_images = document.metadata.get("images", [])

        for index, text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(document.id, index, text)
            metadata = self._inherit_metadata(document, index, text, doc_images)

            chunks.append(Chunk(
                id=chunk_id,
                text=text,
                metadata=metadata,
                source_ref=document.id,
            ))

        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """Generate a deterministic chunk ID.

        Format: {doc_id}_{index:04d}_{hash_8chars}
        """
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    def _inherit_metadata(
        self, document: Document, chunk_index: int, chunk_text: str, doc_images: list[dict]
    ) -> dict:
        """Build chunk metadata: inherit from document + add chunk-specific fields."""
        meta = {}

        # Inherit document-level metadata (except images — distributed per chunk)
        for key, value in document.metadata.items():
            if key != "images":
                meta[key] = value

        # Add chunk-specific fields
        meta["chunk_index"] = chunk_index
        meta["block_type"] = self._classify_block_type(chunk_text)

        # Distribute image references: only include images whose placeholders appear in this chunk
        image_refs = []
        chunk_images = []
        if doc_images:
            for img in doc_images:
                placeholder = f"[IMAGE: {img['id']}]"
                if placeholder in chunk_text:
                    image_refs.append(img["id"])
                    chunk_images.append(img)
                # Also check markdown image syntax
                elif f"![]({img.get('path', '')})" in chunk_text or img.get("path", "") in chunk_text:
                    image_refs.append(img["id"])
                    chunk_images.append(img)

        if image_refs:
            meta["image_refs"] = image_refs
            meta["images"] = chunk_images

        return meta

    # _TABLE_RE 匹配连续两行及以上的 Markdown 表格
    _TABLE_RE = re.compile(r"(?:^\|.+\|\s*$\n?){2,}", re.MULTILINE)
    # _CODE_RE 匹配围栏代码块
    _CODE_RE = re.compile(r"^```", re.MULTILINE)
    # _IMAGE_RE 匹配图片引用占位
    _IMAGE_RE = re.compile(r"\[IMAGE:\s*[^\]]+\]|!\[[^\]]*\]\([^)]+\)")
    # _HEADING_RE 匹配 Markdown 标题
    _HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)

    @classmethod
    def _classify_block_type(cls, text: str) -> str:
        """Classify a chunk's dominant block type for structure-aware metadata.

        Returns one of ``table``/``code``/``image``/``heading``/``text``. Tables and
        code take precedence so structural blocks are not mislabeled as plain text.
        """
        if cls._TABLE_RE.search(text):
            return "table"
        if cls._CODE_RE.search(text):
            return "code"
        if cls._IMAGE_RE.search(text):
            return "image"
        stripped = text.lstrip()
        if cls._HEADING_RE.match(stripped):
            return "heading"
        return "text"
