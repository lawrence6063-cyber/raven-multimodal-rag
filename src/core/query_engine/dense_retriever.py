"""DenseRetriever — semantic vector search using embedding + vector store."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import RetrievalResult
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.libs.embedding.base_embedding import BaseEmbedding
    from src.libs.vector_store.base_vector_store import BaseVectorStore


class DenseRetriever:
    """Retrieves chunks by embedding similarity (dense vector search)."""

    def __init__(
        self,
        settings: "Settings",
        embedding_client: "BaseEmbedding | None" = None,
        vector_store: "BaseVectorStore | None" = None,
    ):
        self._embedding = embedding_client or EmbeddingFactory.create(settings.embedding)
        self._store = vector_store or VectorStoreFactory.create(settings.vector_store)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k chunks by semantic similarity.

        Args:
            query: The search query text.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        # Embed the query
        vectors = self._embedding.embed([query])
        if not vectors or not vectors[0]:
            return []

        query_vector = vectors[0]

        # Query vector store
        results = self._store.query(vector=query_vector, top_k=top_k, filters=filters)

        return [
            RetrievalResult(
                chunk_id=r.id,
                score=r.score,
                text=r.text,
                metadata=r.metadata,
            )
            for r in results
        ]

    def retrieve_by_vector(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve top-k chunks by a precomputed query vector.

        Enables cross-modal retrieval: callers embed an image (or any modality)
        with the same provider and search the shared space directly.

        Args:
            vector: A query vector in the store's embedding space.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of RetrievalResult sorted by score descending.
        """
        if not vector:
            return []

        results = self._store.query(vector=vector, top_k=top_k, filters=filters)
        return [
            RetrievalResult(
                chunk_id=r.id,
                score=r.score,
                text=r.text,
                metadata=r.metadata,
            )
            for r in results
        ]

    def embed_image_query(self, image: str | bytes) -> list[float]:
        """Embed a query image into the shared vector space.

        Args:
            image: Image as a local path, raw bytes, or base64 data URI.

        Returns:
            The image's embedding vector, or an empty list when the provider
            has no image support or returns nothing.
        """
        if not self._embedding.supports_images():
            return []
        vectors = self._embedding.embed_image([image])
        return vectors[0] if vectors and vectors[0] else []
