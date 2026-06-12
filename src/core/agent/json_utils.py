"""JSON parsing helpers for agent LLM outputs.

LLMs frequently wrap JSON in prose or code fences. These helpers extract the
first balanced JSON object/array by slicing between the outermost braces (the
same robust-but-simple strategy used by ``LLMReranker._parse_scores``) and parse
it safely with ``json.loads`` — never ``eval``. On any failure they return
``None`` so callers can apply a conservative fallback.
"""

from __future__ import annotations

import json
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract and parse the outermost JSON object from ``text``.

    Returns:
        The parsed dict, or None when no valid object can be extracted.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def extract_json_array(text: str) -> list[Any] | None:
    """Extract and parse the outermost JSON array from ``text``.

    Returns:
        The parsed list, or None when no valid array can be extracted.
    """
    if not text:
        return None
    start = text.find("[")
    end = text.rfind("]") + 1
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end])
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return data if isinstance(data, list) else None
