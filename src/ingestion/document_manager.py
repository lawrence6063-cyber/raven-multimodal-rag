"""DocumentManager — cross-storage document lifecycle management (G2).

Coordinates ChromaStore, BM25Indexer, ImageStorage, and FileIntegrity to
provide unified list / detail / delete / stats operations for ingested docs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.observability.logger import get_logger

logger = get_logger("ingestion.document_manager")


@dataclass
class DocumentInfo:
    """Summary info for one ingested document."""

    source_path: str
    collection: str
    chunk_count: int
    image_count: int
    file_hash: str = ""
    processed_at: str = ""


@dataclass
class DocumentDetail:
    """Full detail for one document including chunk list."""

    source_path: str
    collection: str
    doc_id: str
    chunk_count: int
    image_count: int
    chunks: list[dict[str, Any]] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DeleteResult:
    """Result of a document deletion."""

    source_path: str
    collection: str
    chunks_deleted: int = 0
    images_deleted: int = 0
    success: bool = True
    error: str = ""


@dataclass
class CollectionStats:
    """Aggregate statistics for a collection (or all collections)."""

    collection: str
    total_documents: int = 0
    total_chunks: int = 0
    total_images: int = 0


class DocumentManager:
    """Manages document lifecycle across all storage backends.

    Provides a unified interface over ChromaStore, BM25Indexer,
    ImageStorage, and FileIntegrity for CRUD operations on documents.
    """

    def __init__(
        self,
        chroma_store,
        bm25_indexer,
        image_storage,
        file_integrity,
    ):
        """Initialize DocumentManager.

        Args:
            chroma_store: ChromaStore instance (with get_by_metadata/delete_by_metadata).
            bm25_indexer: BM25Indexer instance (with remove_document).
            image_storage: ImageStorage instance (with list_images/delete).
            file_integrity: SQLiteIntegrityChecker instance (with list_processed/remove_record).
        """
        self._chroma = chroma_store
        self._bm25 = bm25_indexer
        self._images = image_storage
        self._integrity = file_integrity

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """List all ingested documents, optionally filtered by collection.

        Args:
            collection: Filter by collection name; None = all.

        Returns:
            List of DocumentInfo summaries.
        """
        processed = self._integrity.list_processed()
        all_images = self._images.list_images(collection)

        # Build image count by file_hash (doc_hash in image_index)
        image_counts: dict[str, int] = {}
        for img in all_images:
            dh = img.get("doc_hash", "")
            if dh:
                image_counts[dh] = image_counts.get(dh, 0) + 1

        results: list[DocumentInfo] = []
        for rec in processed:
            file_path = rec.get("file_path", "")
            file_hash = rec.get("file_hash", "")
            chunk_count = rec.get("chunk_count", 0) or 0

            # Determine collection from file_path if possible
            doc_collection = self._guess_collection(file_path)

            if collection and doc_collection != collection:
                continue

            results.append(DocumentInfo(
                source_path=file_path,
                collection=doc_collection,
                chunk_count=chunk_count,
                image_count=image_counts.get(file_hash, 0),
                file_hash=file_hash,
                processed_at=rec.get("processed_at", ""),
            ))

        return results

    def get_document_detail(self, doc_id: str) -> DocumentDetail | None:
        """Get full detail for a document by doc_id.

        Args:
            doc_id: Unique document identifier.

        Returns:
            DocumentDetail or None if not found.
        """
        records = self._chroma.get_by_metadata({"doc_id": doc_id}, limit=1000)
        if not records:
            return None

        chunks = []
        collection = ""
        source_path = ""
        for r in records:
            meta = r.metadata or {}
            if not collection:
                collection = meta.get("collection", "")
            if not source_path:
                source_path = meta.get("source_path", "") or meta.get("file_name", "")
            chunks.append({
                "chunk_id": r.id,
                "text": r.text[:200] if r.text else "",
                "metadata": meta,
            })

        # Get images by doc hash (first chunk's metadata may have it)
        images = []
        first_meta = records[0].metadata or {} if records else {}
        image_refs = first_meta.get("image_refs", [])
        if isinstance(image_refs, list):
            for ref in image_refs:
                path = self._images.get_path(ref)
                if path:
                    images.append({"image_id": ref, "file_path": path})

        return DocumentDetail(
            source_path=source_path,
            collection=collection,
            doc_id=doc_id,
            chunk_count=len(chunks),
            image_count=len(images),
            chunks=chunks,
            images=images,
        )

    def delete_document(self, source_path: str, collection: str = "") -> DeleteResult:
        """Delete a document and all associated data across storages.

        Coordinates deletion in: ChromaStore, BM25Indexer, ImageStorage, FileIntegrity.

        Args:
            source_path: The original file path of the ingested document.
            collection: Optional collection filter.

        Returns:
            DeleteResult with counts and status.
        """
        result = DeleteResult(source_path=source_path, collection=collection)

        try:
            # Find file hash for this source_path
            processed = self._integrity.list_processed()
            file_hash = ""
            for rec in processed:
                if rec.get("file_path", "") == source_path:
                    file_hash = rec.get("file_hash", "")
                    break

            if not file_hash:
                result.success = False
                result.error = f"Document not found in history: {source_path}"
                return result

            # 1. Get chunk IDs from Chroma by source_path metadata
            chroma_records = self._chroma.get_by_metadata(
                {"source_path": source_path}, limit=10000
            )
            chunk_ids = [r.id for r in chroma_records]

            # 2. Delete from Chroma
            if chunk_ids:
                self._chroma.delete(chunk_ids)
                result.chunks_deleted = len(chunk_ids)

            # 3. Delete from BM25
            if chunk_ids:
                self._bm25.remove_document(chunk_ids)

            # 4. Delete images associated with this doc_hash
            images = self._images.list_images()
            for img in images:
                if img.get("doc_hash") == file_hash:
                    self._images.delete(img["image_id"])
                    result.images_deleted += 1

            # 5. Remove from integrity history
            self._integrity.remove_record(file_hash)

            logger.info(
                f"Deleted document: {source_path} "
                f"(chunks={result.chunks_deleted}, images={result.images_deleted})"
            )

        except Exception as e:
            result.success = False
            result.error = str(e)
            logger.error(f"Failed to delete document {source_path}: {e}")

        return result

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """Get aggregate statistics for a collection or all.

        Args:
            collection: Specific collection, or None for totals.

        Returns:
            CollectionStats with totals.
        """
        docs = self.list_documents(collection)
        images = self._images.list_images(collection)

        return CollectionStats(
            collection=collection or "all",
            total_documents=len(docs),
            total_chunks=sum(d.chunk_count for d in docs),
            total_images=len(images),
        )

    @staticmethod
    def _guess_collection(file_path: str) -> str:
        """Try to extract collection name from file path structure."""
        from pathlib import Path

        parts = Path(file_path).parts
        # Convention: data/documents/{collection}/{file}
        for i, part in enumerate(parts):
            if part == "documents" and i + 1 < len(parts) - 1:
                return parts[i + 1]
        return "default"
