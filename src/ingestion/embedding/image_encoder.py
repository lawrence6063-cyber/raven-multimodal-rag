"""ImageEncoder — embeds document images into the shared multimodal space (path B).

For each image extracted from a document, this encoder:
1. Registers the image file in :class:`ImageStorage` (SQLite index + managed copy)
   so the query-time :class:`MultimodalAssembler` can resolve it by id.
2. Embeds the image via the multimodal embedding provider's ``embed_image`` into
   the *same* vector space as text chunks, producing an independent
   :class:`ChunkRecord` (id ``img_<image_id>``, ``metadata.modality == "image"``).

This is what enables true cross-modal retrieval ("以文搜图 / 以图搜图"): a text or
image query encoded by the same provider can directly recall these image records.

Resilience: a missing file, an unsupported provider, or a single failed embedding
never aborts ingestion — the image is logged and skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.types import ChunkRecord
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.ingestion.storage.image_storage import ImageStorage
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings
    from src.core.types import Document

logger = get_logger("ingestion.image_encoder")


class ImageEncoder:
    """Encodes document images into independent multimodal vector records."""

    def __init__(self, settings: "Settings", image_storage: ImageStorage | None = None):
        self._enabled = settings.ingestion.image_embedding
        self._embedding = EmbeddingFactory.create(settings.embedding)
        self._storage = image_storage or ImageStorage()

    def encode_document(
        self,
        document: "Document",
        collection: str = "default",
        captions: dict[str, str] | None = None,
    ) -> list[ChunkRecord]:
        """Store and embed each image of a document as an independent record.

        Args:
            document: The loaded document; reads ``metadata['images']``.
            collection: Collection name images are filed under.
            captions: Optional ``image_id -> caption`` map (woven into record text
                so the image is also discoverable via lexical/textual search).

        Returns:
            List of image ``ChunkRecord`` (possibly empty). Disabled providers or
            no images yield an empty list.
        """
        if not self._enabled:
            return []

        images = document.metadata.get("images", []) or []
        if not images:
            return []

        if not self._embedding.supports_images():
            logger.warning(
                f"Embedding provider '{self._embedding.provider_name}' has no image "
                f"support; skipping image embedding for {document.id}"
            )
            return []

        doc_hash = document.metadata.get("doc_hash", "")
        captions = captions or {}
        records: list[ChunkRecord] = []

        for img in images:
            image_id = img.get("id", "")
            path = img.get("path", "")
            page = int(img.get("page", 0))
            if not image_id or not path:
                continue

            try:
                stored_path = self._storage.save(
                    image_id, path, collection=collection, doc_hash=doc_hash, page_num=page
                )
            except FileNotFoundError:
                logger.warning(f"Image file missing, skipping: {path}")
                continue

            try:
                vector = self._embedding.embed_image([stored_path])[0]
            except Exception as e:  # never abort ingestion on one image
                logger.warning(f"Image embedding failed for {image_id}, skipping: {e}")
                continue

            caption = captions.get(image_id, "")
            text = f"[Image {image_id}] {caption}".strip()
            records.append(
                ChunkRecord(
                    id=f"img_{image_id}",
                    text=text,
                    metadata={
                        "modality": "image",
                        "image_refs": [image_id],
                        "collection": collection,
                        "doc_hash": doc_hash,
                        "page": page,
                        "source_ref": document.id,
                    },
                    dense_vector=vector,
                    sparse_vector={},
                )
            )

        if records:
            logger.info(f"Encoded {len(records)} image vector(s) for {document.id}")
        return records
