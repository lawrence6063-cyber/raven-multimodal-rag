"""DataService — encapsulates data access for the Dashboard data browser (G3)."""

from __future__ import annotations

from typing import Any

from src.core.settings import Settings
from src.ingestion.document_manager import DocumentManager, DocumentInfo, DocumentDetail
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.image_storage import ImageStorage
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.vector_store.vector_store_factory import VectorStoreFactory
from src.observability.logger import get_logger

logger = get_logger("dashboard.data_service")


class DataService:
    """Provides data access for the Dashboard data browser page."""

    def __init__(self, settings: Settings | None = None):
        self._settings = settings
        self._manager: DocumentManager | None = None

    def _get_settings(self) -> Settings:
        if self._settings is None:
            from src.core.settings import load_settings

            self._settings = load_settings()
        return self._settings

    def _get_manager(self) -> DocumentManager:
        if self._manager is None:
            s = self._get_settings()
            chroma = VectorStoreFactory.create(s.vector_store)
            bm25 = BM25Indexer(s.ingestion.bm25_index_path)
            images = ImageStorage()
            integrity = SQLiteIntegrityChecker()
            self._manager = DocumentManager(chroma, bm25, images, integrity)
        return self._manager

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """List ingested documents."""
        return self._get_manager().list_documents(collection)

    def get_document_detail(self, doc_id: str) -> DocumentDetail | None:
        """Get full detail for a document."""
        return self._get_manager().get_document_detail(doc_id)

    def list_collections(self) -> list[str]:
        """List available collections from integrity history."""
        docs = self._get_manager().list_documents()
        return sorted({d.collection for d in docs})

    def get_chunk_details(self, doc_id: str) -> list[dict[str, Any]]:
        """Get chunk list for a document."""
        detail = self.get_document_detail(doc_id)
        if detail:
            return detail.chunks
        return []
