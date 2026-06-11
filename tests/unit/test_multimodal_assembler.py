"""Unit tests for MultimodalAssembler (E6)."""

from __future__ import annotations

import base64

import pytest

from src.core.response.multimodal_assembler import MultimodalAssembler
from src.core.types import RetrievalResult


class _FakeStorage:
    """In-memory stand-in for ImageStorage.get_path."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def get_path(self, image_id: str):
        return self._mapping.get(image_id)


@pytest.fixture
def png_file(tmp_path):
    """A tiny valid-ish PNG file on disk."""
    path = tmp_path / "img1.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n fake-bytes")
    return path


def _result(image_refs):
    return RetrievalResult(
        chunk_id="c", score=0.5, text="t", metadata={"image_refs": image_refs}
    )


def test_assemble_encodes_known_image(png_file):
    storage = _FakeStorage({"img1": str(png_file)})
    contents = MultimodalAssembler(storage).assemble([_result(["img1"])])
    assert len(contents) == 1
    assert contents[0].type == "image"
    assert contents[0].mime_type == "image/png"
    assert base64.b64decode(contents[0].data) == png_file.read_bytes()


def test_deduplicates_image_ids(png_file):
    storage = _FakeStorage({"img1": str(png_file)})
    results = [_result(["img1"]), _result(["img1"])]
    assert len(MultimodalAssembler(storage).assemble(results)) == 1


def test_missing_id_is_skipped():
    storage = _FakeStorage({})
    assert MultimodalAssembler(storage).assemble([_result(["ghost"])]) == []


def test_oversized_image_is_skipped(png_file):
    storage = _FakeStorage({"img1": str(png_file)})
    assembler = MultimodalAssembler(storage, max_bytes=1)
    assert assembler.assemble([_result(["img1"])]) == []


def test_no_image_refs_returns_empty():
    storage = _FakeStorage({})
    result = RetrievalResult(chunk_id="c", score=0.5, text="t", metadata={})
    assert MultimodalAssembler(storage).assemble([result]) == []


def test_jpeg_mime_resolved(tmp_path):
    path = tmp_path / "p.jpeg"
    path.write_bytes(b"\xff\xd8\xff fake-jpeg")
    storage = _FakeStorage({"j": str(path)})
    contents = MultimodalAssembler(storage).assemble([_result(["j"])])
    assert contents[0].mime_type == "image/jpeg"
