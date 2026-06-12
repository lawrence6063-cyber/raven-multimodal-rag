"""Ingestion Pipeline — orchestrates the full document-to-vectors flow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, TYPE_CHECKING

from src.core.types import Document, Chunk, ChunkRecord
from src.core.trace.trace_collector import TraceCollector
from src.core.trace.trace_context import TraceContext
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.libs.loader.loader_factory import LoaderFactory
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.embedding.image_encoder import ImageEncoder
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import Settings

logger = get_logger("ingestion.pipeline")


class IngestionPipeline:
    """Orchestrates the full ingestion flow: file → vectors + BM25 index."""

    def __init__(self, settings: "Settings"):
        self._settings = settings
        self._integrity = SQLiteIntegrityChecker()
        self._loader = LoaderFactory.create(settings.loader)
        self._loader_provider = getattr(settings.loader, "provider", "markitdown")
        self._chunker = DocumentChunker(settings)
        self._refiner = ChunkRefiner(settings)
        self._enricher = MetadataEnricher(settings)
        self._captioner = ImageCaptioner(settings)
        self._batch_processor = BatchProcessor(settings)
        self._image_encoder = ImageEncoder(settings)
        self._bm25 = BM25Indexer(settings.ingestion.bm25_index_path)
        self._upserter = VectorUpserter(settings)
        self._collector = TraceCollector(settings)

    def run(
        self,
        file_path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: Callable[[str, int, int], None] | None = None,
        trace: TraceContext | None = None,
    ) -> dict:
        """Run the full ingestion pipeline for a single file.

        Args:
            file_path: Path to the document file.
            collection: Collection name for organization.
            force: If True, skip integrity check and re-process.
            on_progress: Optional callback(stage_name, current, total).
            trace: Optional TraceContext; an ingestion trace is created when None.

        Returns:
            Summary dict with chunk_count, status, etc.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        own_trace = trace is None
        if trace is None:
            trace = TraceContext(trace_type="ingestion")

        # Stage 1: Integrity check
        if on_progress:
            on_progress("integrity_check", 0, 1)

        file_hash = self._integrity.compute_sha256(path)
        if not force and self._integrity.should_skip(file_hash):
            logger.info(f"Skipping (already processed): {file_path}")
            return {"status": "skipped", "file": str(path), "reason": "already_processed"}

        embed_method = getattr(self._settings.embedding, "provider", "embedding")

        try:
            # Stage 2: Load
            if on_progress:
                on_progress("load", 0, 1)
            logger.info(f"Loading: {file_path}")
            start = time.perf_counter()
            document = self._loader.load(str(path))
            document.metadata["collection"] = collection
            self._trace_stage(
                trace, "load", start, method=getattr(self, "_loader_provider", "markitdown")
            )

            # Stage 3: Split
            if on_progress:
                on_progress("split", 0, 1)
            logger.info("Splitting into chunks...")
            start = time.perf_counter()
            chunks = self._chunker.split_document(document)
            logger.info(f"Produced {len(chunks)} chunks")
            self._trace_stage(
                trace, "split", start,
                method=getattr(self._settings.splitter, "provider", "recursive"),
                chunks=len(chunks),
            )

            # Stage 4: Transform
            if on_progress:
                on_progress("transform", 0, 3)
            logger.info("Refining chunks...")
            start = time.perf_counter()
            chunks = self._refiner.transform(chunks)
            if on_progress:
                on_progress("transform", 1, 3)

            logger.info("Enriching metadata...")
            chunks = self._enricher.transform(chunks)
            if on_progress:
                on_progress("transform", 2, 3)

            logger.info("Processing images...")
            chunks = self._captioner.transform(chunks)
            if on_progress:
                on_progress("transform", 3, 3)
            self._trace_stage(
                trace, "transform", start,
                method="refine+enrich+caption", chunks=len(chunks),
            )

            # Stage 5: Encode (embed)
            if on_progress:
                on_progress("encode", 0, 1)
            logger.info("Encoding chunks (dense + sparse)...")
            start = time.perf_counter()
            records = self._batch_processor.process(chunks)

            # Stage 5b: Encode document images into the shared multimodal space
            captions = self._collect_captions(chunks)
            image_records = self._image_encoder.encode_document(
                document, collection=collection, captions=captions
            )
            if image_records:
                records.extend(image_records)
                logger.info(f"Added {len(image_records)} image vector record(s)")
            self._trace_stage(
                trace, "embed", start, method=embed_method, records=len(records),
                image_records=len(image_records),
            )

            # Stage 6: Store (upsert)
            if on_progress:
                on_progress("store", 0, 2)
            logger.info("Upserting to vector store...")
            start = time.perf_counter()
            self._upserter.upsert(records)
            if on_progress:
                on_progress("store", 1, 2)

            logger.info("Building BM25 index...")
            self._bm25.build(records)
            if on_progress:
                on_progress("store", 2, 2)
            self._trace_stage(
                trace, "upsert", start, method="chroma+bm25", records=len(records),
            )

            # Mark success
            self._integrity.mark_success(file_hash, str(path), chunk_count=len(records))

            result = {
                "status": "success",
                "file": str(path),
                "collection": collection,
                "chunk_count": len(records),
                "doc_id": document.id,
            }
            logger.info(f"Ingestion complete: {len(records)} chunks stored")
            return result

        except Exception as e:
            self._integrity.mark_failed(file_hash, str(e))
            logger.error(f"Ingestion failed for {file_path}: {e}")
            if trace is not None:
                trace.record_stage("error", method=type(e).__name__)
            raise

        finally:
            if own_trace:
                trace.finish()
                self._collector.collect(trace)

    @staticmethod
    def _collect_captions(chunks: list[Chunk]) -> dict[str, str]:
        """Aggregate image_id -> caption from chunk metadata produced by captioning."""
        captions: dict[str, str] = {}
        for chunk in chunks:
            for image_id, caption in (chunk.metadata.get("image_captions") or {}).items():
                if caption and not captions.get(image_id):
                    captions[image_id] = caption
        return captions

    @staticmethod
    def _trace_stage(
        trace: TraceContext | None,
        name: str,
        start: float,
        method: str = "",
        **details,
    ) -> None:
        """Record an ingestion stage on the trace (if any)."""
        if trace is None:
            return
        elapsed = (time.perf_counter() - start) * 1000.0
        trace.record_stage(name, method=method, elapsed_ms=elapsed, **details)
