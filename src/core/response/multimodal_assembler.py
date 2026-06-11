"""MultimodalAssembler — turns image_refs in results into ImageContent (E6).

When a retrieved chunk's metadata carries ``image_refs`` (a list of image ids
produced during ingestion), this assembler resolves each id to a local file via
``ImageStorage`` (index lookup only — never an arbitrary caller-supplied path),
reads the bytes, and Base64-encodes them into ImageContent items.

Reliability/security:
- Paths come exclusively from the ImageStorage index (prevents path traversal).
- Oversized files are skipped to bound response size.
- Any per-image failure is logged and skipped; it never blocks the text answer.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.response.mcp_types import ImageContent
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.types import RetrievalResult
    from src.ingestion.storage.image_storage import ImageStorage

logger = get_logger("response.multimodal_assembler")

# _MAX_IMAGE_BYTES upper bound on a single image's raw size before Base64
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MiB
# _MIME_BY_SUFFIX maps file extensions to MIME types for ImageContent
_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


class MultimodalAssembler:
    """Resolves chunk image references into Base64 ImageContent items."""

    def __init__(self, image_storage: "ImageStorage", max_bytes: int = _MAX_IMAGE_BYTES):
        self._storage = image_storage
        self._max_bytes = max_bytes

    def assemble(self, results: list["RetrievalResult"]) -> list[ImageContent]:
        """Collect ImageContent for all unique image_refs across results.

        Args:
            results: Ranked retrieval results whose metadata may hold image_refs.

        Returns:
            List of ImageContent (possibly empty). De-duplicates by image id
            while preserving first-seen order.
        """
        contents: list[ImageContent] = []
        seen: set[str] = set()

        for result in results:
            image_refs = (result.metadata or {}).get("image_refs") or []
            for image_id in image_refs:
                if not image_id or image_id in seen:
                    continue
                seen.add(image_id)
                content = self._encode_image(image_id)
                if content is not None:
                    contents.append(content)

        return contents

    def _encode_image(self, image_id: str) -> ImageContent | None:
        """Resolve, validate, and Base64-encode a single image; None on failure."""
        try:
            path_str = self._storage.get_path(image_id)
            if not path_str:
                logger.warning(f"Image id not in index, skipping: {image_id}")
                return None

            path = Path(path_str)
            if not path.is_file():
                logger.warning(f"Image file missing on disk, skipping: {path_str}")
                return None

            size = path.stat().st_size
            if size > self._max_bytes:
                logger.warning(
                    f"Image exceeds size limit ({size} > {self._max_bytes}), "
                    f"skipping: {image_id}"
                )
                return None

            data = base64.b64encode(path.read_bytes()).decode("ascii")
            mime_type = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
            return ImageContent(data=data, mime_type=mime_type)

        except Exception as e:  # never block the text answer
            logger.warning(f"Failed to encode image {image_id}, skipping: {e}")
            return None
