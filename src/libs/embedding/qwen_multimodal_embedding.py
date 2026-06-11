"""QwenMultimodalEmbedding — DashScope multimodal embedding (text + image).

Encodes both text and images into the *same* vector space via Alibaba Cloud
DashScope's multimodal embedding models (e.g. ``multimodal-embedding-v1``),
enabling cross-modal retrieval (text↔image) for the RAG pipeline.

Design notes / why per-item calls:
- Unlike OpenAI-compatible text embedding, the DashScope multimodal endpoint is
  **not** OpenAI-compatible and must be called through the ``dashscope`` SDK.
- Some multimodal models *fuse* a multi-element ``input`` into a single vector.
  To guarantee one independent vector per chunk/image (the RAG contract) and to
  stay compatible across model variants, each text/image is embedded with its
  own single-element call.
- ``auto_truncation`` is enabled so long passages do not fail the request.

The concrete model name and dimension are fully configurable through
``EmbeddingSettings`` so the provider can target stronger multimodal models
without code changes.
"""

from __future__ import annotations

import base64
import binascii
import os
import random
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING

from src.libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from src.libs.embedding.embedding_factory import register_embedding
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import EmbeddingSettings

logger = get_logger("embedding.qwen_multimodal")


@register_embedding("qwen_multimodal")
class QwenMultimodalEmbedding(BaseEmbedding):
    """DashScope multimodal embedding placing text and images in one space."""

    # DEFAULT_MODEL default DashScope multimodal embedding model
    DEFAULT_MODEL = "multimodal-embedding-v1"
    # DEFAULT_DIMENSIONS fallback vector dimension when settings omit it
    DEFAULT_DIMENSIONS = 1024
    # MAX_TEXT_CHARS hard cap below the model's 10240-char text limit (server-side
    # auto_truncation is unreliable, so we truncate client-side as a safety net)
    MAX_TEXT_CHARS = 8000
    # MAX_RETRIES retry attempts on throttling / transient errors
    MAX_RETRIES = 6
    # RETRY_BASE_DELAY base seconds for exponential backoff (2,4,8,16,...)
    RETRY_BASE_DELAY = 2.0

    def __init__(self, settings: "EmbeddingSettings"):
        self._settings = settings
        self._model = settings.model or self.DEFAULT_MODEL
        self._api_key = settings.api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._dimensions = settings.dimensions or self.DEFAULT_DIMENSIONS

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts into the shared multimodal vector space.

        Args:
            texts: Text strings to encode.

        Returns:
            One vector per input text, in input order.

        Raises:
            EmbeddingError: If any API call fails.
        """
        if not texts:
            return []
        return [self._embed_one({"text": self._prepare_text(text)}) for text in texts]

    def embed_image(self, images: list[str | bytes]) -> list[list[float]]:
        """Embed images into the same vector space as text.

        Args:
            images: Each image as a local path, raw bytes, base64 data URI, or
                a public URL.

        Returns:
            One vector per input image, in input order.

        Raises:
            EmbeddingError: If any API call fails.
        """
        if not images:
            return []
        vectors: list[list[float]] = []
        for image in images:
            value, tmp_path = self._prepare_image_input(image)
            try:
                vectors.append(self._embed_one({"image": value}))
            finally:
                if tmp_path is not None:
                    Path(tmp_path).unlink(missing_ok=True)
        return vectors

    def _embed_one(self, content: dict[str, str]) -> list[float]:
        """Call DashScope for a single content item and return its vector.

        Retries with exponential backoff on throttling / transient failures so a
        long ingest (hundreds of per-item calls) survives DashScope rate limits.
        """
        from http import HTTPStatus

        try:
            from dashscope import MultiModalEmbedding
        except ImportError as e:  # pragma: no cover - dependency guard
            raise EmbeddingError(
                "dashscope SDK not installed. Run: pip install dashscope",
                provider="qwen_multimodal",
                cause=e,
            ) from e

        kwargs: dict = {"model": self._model, "input": [content]}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        parameters: dict = {"auto_truncation": True}
        if self._dimensions:
            parameters["dimension"] = self._dimensions
        kwargs["parameters"] = parameters

        last_error = ""
        for attempt in range(self.MAX_RETRIES + 1):
            from src.libs.throttle import dashscope_limiter

            dashscope_limiter.wait()  # proactive throttle to stay under QPS limit
            try:
                resp = MultiModalEmbedding.call(**kwargs)
            except Exception as e:  # network/transient — retry then give up
                last_error = str(e)
                if attempt < self.MAX_RETRIES:
                    time.sleep(self._backoff(attempt))
                    continue
                raise EmbeddingError(
                    f"DashScope multimodal embedding call failed: {e}",
                    provider="qwen_multimodal",
                    cause=e,
                ) from e

            if getattr(resp, "status_code", None) == HTTPStatus.OK:
                embeddings = (resp.output or {}).get("embeddings") or []
                if not embeddings:
                    raise EmbeddingError(
                        "DashScope multimodal embedding returned no vectors",
                        provider="qwen_multimodal",
                    )
                return embeddings[0]["embedding"]

            code = str(getattr(resp, "code", "") or "")
            message = str(getattr(resp, "message", "") or "")
            last_error = f"code={code}, message={message}"

            if self._is_throttling(code, message) and attempt < self.MAX_RETRIES:
                delay = self._backoff(attempt)
                logger.warning(f"DashScope throttled, retry in {delay:.1f}s ({attempt + 1}/{self.MAX_RETRIES})")
                time.sleep(delay)
                continue

            raise EmbeddingError(
                f"DashScope multimodal embedding failed: {last_error}",
                provider="qwen_multimodal",
            )

        raise EmbeddingError(
            f"DashScope multimodal embedding failed after {self.MAX_RETRIES} retries: {last_error}",
            provider="qwen_multimodal",
        )

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter (seconds)."""
        return self.RETRY_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1.0)

    @staticmethod
    def _is_throttling(code: str, message: str) -> bool:
        """Whether the response indicates a retriable rate-limit/throttle."""
        blob = f"{code} {message}".lower()
        return "throttl" in blob or "rate limit" in blob or "ratequota" in blob

    def _prepare_text(self, text: str) -> str:
        """Truncate text to the model's safe limit (a single space if empty)."""
        if not text:
            return " "
        if len(text) > self.MAX_TEXT_CHARS:
            logger.warning(
                f"Text length {len(text)} exceeds {self.MAX_TEXT_CHARS}, truncating"
            )
            return text[: self.MAX_TEXT_CHARS]
        return text

    def _prepare_image_input(self, image: str | bytes) -> tuple[str, str | None]:
        """Normalise an image into a DashScope-acceptable reference.

        Returns a ``(value, tmp_path)`` pair where ``value`` is passed to the
        API and ``tmp_path`` (when not None) must be deleted by the caller.

        - Public URLs are forwarded as-is.
        - Local paths become ``file://<abs>`` references.
        - Raw bytes / base64 data URIs are written to a temp file and returned
          as ``file://`` references (the endpoint does not accept inline base64).
        """
        if isinstance(image, bytes):
            return self._write_temp(image)

        text = str(image)
        if text.startswith(("http://", "https://")):
            return text, None
        if text.startswith("data:"):
            payload = text.split(",", 1)[-1]
            return self._write_temp(self._decode_base64(payload))

        path = Path(text)
        if path.is_file():
            return f"file://{path.resolve()}", None

        # Fall back to treating the string as raw base64.
        return self._write_temp(self._decode_base64(text))

    @staticmethod
    def _decode_base64(payload: str) -> bytes:
        """Decode a base64 payload, wrapping errors as EmbeddingError."""
        try:
            return base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as e:
            raise EmbeddingError(
                "Invalid image input: not a valid path or base64 data",
                provider="qwen_multimodal",
                cause=e,
            ) from e

    @staticmethod
    def _write_temp(data: bytes) -> tuple[str, str]:
        """Write image bytes to a temp PNG and return (file:// uri, path)."""
        fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="qwen_mm_")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        return f"file://{tmp_path}", tmp_path

    @property
    def provider_name(self) -> str:
        return "qwen_multimodal"

    @property
    def dimensions(self) -> int:
        return self._dimensions
