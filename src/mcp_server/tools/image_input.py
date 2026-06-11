"""Query image input validation — guards the MCP ``image`` argument.

Accepts only two safe shapes and rejects everything else:
1. Base64 image data (raw or ``data:image/...;base64,`` URI), size-bounded.
2. A local file path confined to a whitelisted base directory (default ``data/``),
   blocking path traversal and absolute escapes.

Remote URLs (``http(s)://``) are rejected outright to avoid SSRF.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path

# _MAX_IMAGE_BYTES upper bound on a decoded query image (10 MiB)
_MAX_IMAGE_BYTES = 10 * 1024 * 1024


class ImageInputError(ValueError):
    """Raised when a query image input is malformed or disallowed."""


def validate_query_image(image: str, allowed_base_dir: str = "data") -> str | bytes:
    """Validate and normalise a query image argument.

    Args:
        image: A base64 string / ``data:`` URI, or a local file path under
            ``allowed_base_dir``.
        allowed_base_dir: Directory local paths must resolve inside.

    Returns:
        Raw image bytes (for base64 inputs) or a safe absolute path string.

    Raises:
        ImageInputError: If the input is empty, a remote URL, escapes the
            whitelist, is missing, or exceeds the size limit.
    """
    if not image or not isinstance(image, str):
        raise ImageInputError("image must be a non-empty string")

    text = image.strip()

    if text.startswith(("http://", "https://", "ftp://")):
        raise ImageInputError("remote image URLs are not allowed")

    if text.startswith("data:"):
        payload = text.split(",", 1)[-1]
        return _decode_checked(payload)

    # Treat as a local path first; if it is not a real file, fall back to base64.
    candidate = Path(text)
    looks_like_path = candidate.suffix != "" and "\n" not in text and len(text) < 4096
    if looks_like_path:
        base = Path(allowed_base_dir).resolve()
        resolved = candidate.resolve()
        try:
            resolved.relative_to(base)
        except ValueError:
            raise ImageInputError(
                f"image path must be inside '{allowed_base_dir}/'"
            )
        if not resolved.is_file():
            raise ImageInputError(f"image file not found: {text}")
        if resolved.stat().st_size > _MAX_IMAGE_BYTES:
            raise ImageInputError("image file exceeds the size limit")
        return str(resolved)

    # Otherwise assume raw base64.
    return _decode_checked(text)


def _decode_checked(payload: str) -> bytes:
    """Decode base64 and enforce the size limit."""
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ImageInputError("image is neither a valid path nor base64 data") from e
    if not data:
        raise ImageInputError("decoded image is empty")
    if len(data) > _MAX_IMAGE_BYTES:
        raise ImageInputError("image exceeds the size limit")
    return data
