"""ChromaStore — Chroma vector database implementation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.libs.vector_store.base_vector_store import (
    BaseVectorStore,
    VectorRecord,
    QueryResult,
    VectorStoreError,
)
from src.libs.vector_store.vector_store_factory import register_vector_store

if TYPE_CHECKING:
    from src.core.settings import VectorStoreSettings


@register_vector_store("chroma")
class ChromaStore(BaseVectorStore):
    """ChromaDB vector store implementation with local persistence."""

    def __init__(self, settings: "VectorStoreSettings"):
        self._collection_name = settings.collection_name
        self._persist_directory = settings.persist_directory
        self._collection = None
        self._client = None

    def _get_collection(self):
        """Lazy-initialize Chroma client and collection."""
        if self._collection is None:
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                self._client = chromadb.PersistentClient(
                    path=self._persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self._collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except ImportError:
                raise VectorStoreError(
                    "chromadb not installed. Run: pip install chromadb",
                    provider="chroma",
                )
            except Exception as e:
                raise VectorStoreError(f"Failed to initialize Chroma: {e}", provider="chroma") from e
        return self._collection

    def upsert(self, records: list[VectorRecord]) -> None:
        """Upsert records into Chroma collection."""
        if not records:
            return

        collection = self._get_collection()
        try:
            collection.upsert(
                ids=[r.id for r in records],
                embeddings=[r.vector for r in records],
                documents=[r.text for r in records],
                metadatas=[self._sanitize_metadata(r.metadata) for r in records],
            )
        except Exception as e:
            raise VectorStoreError(f"Upsert failed: {e}", provider="chroma") from e

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Coerce metadata into Chroma-acceptable values.

        Chroma metadata values must be scalars (str/int/float/bool) or non-empty
        lists of scalars. This sanitizer therefore:
          - drops ``None`` and empty containers (e.g. ``tags=[]`` which Chroma
            rejects with "list value ... to be non-empty");
          - keeps non-empty scalar lists as-is (e.g. ``image_refs`` must stay a
            list so the multimodal assembler can iterate ids after retrieval);
          - JSON-encodes complex values (dicts, lists of dicts such as
            ``images``) to a string so they never break the upsert.

        Args:
            metadata: Raw chunk metadata (may be ``None``).

        Returns:
            A new dict safe to pass to Chroma.
        """
        if not metadata:
            return {}

        clean: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, tuple)):
                if not value:
                    continue  # Chroma rejects empty lists
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    clean[key] = list(value)
                else:
                    clean[key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                if value:
                    clean[key] = json.dumps(value, ensure_ascii=False)
            else:
                clean[key] = str(value)
        return clean

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        """Query Chroma for similar vectors."""
        collection = self._get_collection()
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [vector],
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if filters:
                kwargs["where"] = filters

            results = collection.query(**kwargs)

            query_results = []
            if results and results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results.get("distances") else [0.0] * len(ids)
                documents = results["documents"][0] if results.get("documents") else [""] * len(ids)
                metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)

                for i, id_ in enumerate(ids):
                    # Chroma returns distances; convert to similarity score
                    score = 1.0 - distances[i] if distances[i] <= 1.0 else 0.0
                    query_results.append(QueryResult(
                        id=id_,
                        score=score,
                        text=documents[i] or "",
                        metadata=metadatas[i] or {},
                    ))

            return query_results

        except Exception as e:
            raise VectorStoreError(f"Query failed: {e}", provider="chroma") from e

    def delete(self, ids: list[str]) -> None:
        """Delete records by ID."""
        if not ids:
            return
        collection = self._get_collection()
        try:
            collection.delete(ids=ids)
        except Exception as e:
            raise VectorStoreError(f"Delete failed: {e}", provider="chroma") from e

    def get_by_ids(self, ids: list[str]) -> list[QueryResult]:
        """按 ID 批量获取记录的文本和元数据。

        Args:
            ids: 要获取的记录 ID 列表。

        Returns:
            找到的记录列表（score=0.0），不存在的 ID 会被跳过。
        """
        if not ids:
            return []
        collection = self._get_collection()
        try:
            results = collection.get(ids=ids, include=["documents", "metadatas"])
            records: list[QueryResult] = []
            if results and results["ids"]:
                for i, id_ in enumerate(results["ids"]):
                    records.append(QueryResult(
                        id=id_,
                        score=0.0,
                        text=results["documents"][i] if results.get("documents") else "",
                        metadata=results["metadatas"][i] if results.get("metadatas") else {},
                    ))
            return records
        except Exception as e:
            raise VectorStoreError(f"Get by IDs failed: {e}", provider="chroma") from e

    def get_by_metadata(
        self, where: dict[str, Any], limit: int = 1
    ) -> list[QueryResult]:
        """按 metadata 过滤批量获取记录（不涉及相似度计算）。

        用于按 doc_id 等元数据字段查找记录，例如获取文档摘要。

        Args:
            where: Chroma metadata 过滤条件，如 {"doc_id": "xxx"}。
            limit: 返回记录数上限。

        Returns:
            匹配的记录列表（score=0.0），无匹配时返回空列表。
        """
        if not where:
            return []
        collection = self._get_collection()
        try:
            results = collection.get(
                where=where,
                limit=limit,
                include=["documents", "metadatas"],
            )
            records: list[QueryResult] = []
            if results and results.get("ids"):
                for i, id_ in enumerate(results["ids"]):
                    records.append(QueryResult(
                        id=id_,
                        score=0.0,
                        text=results["documents"][i] if results.get("documents") else "",
                        metadata=results["metadatas"][i] if results.get("metadatas") else {},
                    ))
            return records
        except Exception as e:
            raise VectorStoreError(f"Get by metadata failed: {e}", provider="chroma") from e

    def delete_by_metadata(self, where: dict[str, Any]) -> int:
        """Delete records matching metadata filters.

        Args:
            where: Chroma metadata filter, e.g. {"doc_id": "xxx"}.

        Returns:
            Number of records deleted.
        """
        if not where:
            return 0
        collection = self._get_collection()
        try:
            # Get matching IDs first
            results = collection.get(where=where, include=[])
            ids = results.get("ids", []) if results else []
            if ids:
                collection.delete(ids=ids)
            return len(ids)
        except Exception as e:
            raise VectorStoreError(f"Delete by metadata failed: {e}", provider="chroma") from e

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection statistics.

        Returns:
            Dict with count, collection_name, and persist_directory.
        """
        collection = self._get_collection()
        try:
            count = collection.count()
            return {
                "collection_name": self._collection_name,
                "persist_directory": self._persist_directory,
                "total_chunks": count,
            }
        except Exception as e:
            raise VectorStoreError(f"Get collection stats failed: {e}", provider="chroma") from e

    @property
    def provider_name(self) -> str:
        return "chroma"
