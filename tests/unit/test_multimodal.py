"""Unit tests for the Qwen multimodal capability (path B).

Covers, without any network calls (DashScope/OpenAI are stubbed):
- BaseEmbedding image-support contract.
- QwenMultimodalEmbedding text/image encoding via a fake ``dashscope`` SDK.
- VisionLLMFactory registration and lookup.
- Query-image input validation (security: traversal / SSRF / size).
- ImageEncoder document image embedding into independent records.
- DenseRetriever.retrieve_by_vector / embed_image_query.
- HybridSearch image-only (dense-only) routing.
- QueryKnowledgeHubTool image argument handling.
"""

from __future__ import annotations

import base64
import sys
from types import SimpleNamespace

import pytest

from src.core.settings import EmbeddingSettings, Settings, VisionLLMSettings


# --------------------------------------------------------------------------- #
# Fake dashscope SDK
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, vector):
        self.status_code = 200
        self.code = ""
        self.message = ""
        self.output = {"embeddings": [{"embedding": vector}]}


def _install_fake_dashscope(monkeypatch, captured: list):
    """Install a fake ``dashscope`` module exposing MultiModalEmbedding.call."""
    from http import HTTPStatus

    class _FakeMME:
        @staticmethod
        def call(**kwargs):
            captured.append(kwargs)
            return _FakeResp([0.1, 0.2, 0.3])

    fake = SimpleNamespace(MultiModalEmbedding=_FakeMME)
    monkeypatch.setitem(sys.modules, "dashscope", fake)
    # Ensure status comparison matches our fake 200
    assert HTTPStatus.OK == 200


# --------------------------------------------------------------------------- #
# BaseEmbedding contract
# --------------------------------------------------------------------------- #
def test_text_provider_has_no_image_support():
    from src.libs.embedding.qwen_embedding import QwenEmbedding

    emb = QwenEmbedding(EmbeddingSettings(provider="qwen", model="text-embedding-v3", api_key="k"))
    assert emb.supports_images() is False
    with pytest.raises(NotImplementedError):
        emb.embed_image(["x"])


def test_multimodal_provider_supports_images():
    from src.libs.embedding.qwen_multimodal_embedding import QwenMultimodalEmbedding

    emb = QwenMultimodalEmbedding(EmbeddingSettings(provider="qwen_multimodal", api_key="k"))
    assert emb.supports_images() is True
    assert emb.provider_name == "qwen_multimodal"


# --------------------------------------------------------------------------- #
# QwenMultimodalEmbedding
# --------------------------------------------------------------------------- #
def test_multimodal_embed_text(monkeypatch):
    from src.libs.embedding.qwen_multimodal_embedding import QwenMultimodalEmbedding

    captured: list = []
    _install_fake_dashscope(monkeypatch, captured)

    emb = QwenMultimodalEmbedding(
        EmbeddingSettings(provider="qwen_multimodal", model="multimodal-embedding-v1",
                          api_key="k", dimensions=3)
    )
    vectors = emb.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    # one call per text (independent vectors), with text content + auto_truncation
    assert len(captured) == 2
    assert captured[0]["input"] == [{"text": "hello"}]
    assert captured[0]["parameters"]["auto_truncation"] is True
    assert captured[0]["parameters"]["dimension"] == 3


def test_multimodal_embed_image_from_path(monkeypatch, tmp_path):
    from src.libs.embedding.qwen_multimodal_embedding import QwenMultimodalEmbedding

    captured: list = []
    _install_fake_dashscope(monkeypatch, captured)

    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n fake")

    emb = QwenMultimodalEmbedding(EmbeddingSettings(provider="qwen_multimodal", api_key="k"))
    vectors = emb.embed_image([str(img)])

    assert vectors == [[0.1, 0.2, 0.3]]
    assert captured[0]["input"][0]["image"].startswith("file://")


def test_multimodal_embed_image_from_base64(monkeypatch):
    from src.libs.embedding.qwen_multimodal_embedding import QwenMultimodalEmbedding

    captured: list = []
    _install_fake_dashscope(monkeypatch, captured)

    payload = base64.b64encode(b"\x89PNG\r\n\x1a\n fake").decode()
    emb = QwenMultimodalEmbedding(EmbeddingSettings(provider="qwen_multimodal", api_key="k"))
    vectors = emb.embed_image([f"data:image/png;base64,{payload}"])

    assert vectors == [[0.1, 0.2, 0.3]]
    # temp file path is forwarded as a file:// reference
    assert captured[0]["input"][0]["image"].startswith("file://")


# --------------------------------------------------------------------------- #
# VisionLLMFactory
# --------------------------------------------------------------------------- #
def test_vision_factory_registers_builtins():
    from src.libs.llm.vision_llm_factory import VisionLLMFactory

    providers = VisionLLMFactory.available_providers()
    assert "qwen_vision" in providers
    assert "azure_vision" in providers


def test_vision_factory_creates_qwen_vision():
    from src.libs.llm.vision_llm_factory import VisionLLMFactory
    from src.libs.llm.qwen_vision_llm import QwenVisionLLM

    llm = VisionLLMFactory.create(
        VisionLLMSettings(enabled=True, provider="qwen_vision", model="qwen-vl-max", api_key="k")
    )
    assert isinstance(llm, QwenVisionLLM)
    assert llm.provider_name == "qwen_vision"


def test_vision_factory_unknown_provider_raises():
    from src.libs.llm.vision_llm_factory import VisionLLMFactory
    from src.libs.llm.base_llm import LLMError

    with pytest.raises(LLMError, match="Unknown vision LLM provider"):
        VisionLLMFactory.create(VisionLLMSettings(provider="nope"))


# --------------------------------------------------------------------------- #
# Query image input validation
# --------------------------------------------------------------------------- #
def test_validate_image_base64_ok():
    from src.mcp_server.tools.image_input import validate_query_image

    payload = base64.b64encode(b"hello-image").decode()
    out = validate_query_image(payload)
    assert out == b"hello-image"


def test_validate_image_data_uri_ok():
    from src.mcp_server.tools.image_input import validate_query_image

    payload = base64.b64encode(b"abc").decode()
    out = validate_query_image(f"data:image/png;base64,{payload}")
    assert out == b"abc"


def test_validate_image_rejects_http():
    from src.mcp_server.tools.image_input import validate_query_image, ImageInputError

    with pytest.raises(ImageInputError, match="remote"):
        validate_query_image("http://internal/secret.png")


def test_validate_image_rejects_traversal(tmp_path):
    from src.mcp_server.tools.image_input import validate_query_image, ImageInputError

    with pytest.raises(ImageInputError, match="inside"):
        validate_query_image("../../etc/passwd.png", allowed_base_dir=str(tmp_path))


def test_validate_image_path_ok(tmp_path):
    from src.mcp_server.tools.image_input import validate_query_image

    base = tmp_path / "data"
    base.mkdir()
    img = base / "fig.png"
    img.write_bytes(b"\x89PNG fake")
    out = validate_query_image(str(img), allowed_base_dir=str(base))
    assert out == str(img.resolve())


def test_validate_image_missing_file(tmp_path):
    from src.mcp_server.tools.image_input import validate_query_image, ImageInputError

    with pytest.raises(ImageInputError, match="not found"):
        validate_query_image(str(tmp_path / "nope.png"), allowed_base_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# ImageEncoder
# --------------------------------------------------------------------------- #
class _FakeImageEmbedding:
    provider_name = "fake_mm"

    def embed(self, texts):
        return [[0.0] * 3 for _ in texts]

    def embed_image(self, images):
        return [[1.0, 2.0, 3.0] for _ in images]

    def supports_images(self):
        return True


class _FakeStorage:
    def __init__(self):
        self.saved = []

    def save(self, image_id, source_path, collection="default", doc_hash="", page_num=0):
        self.saved.append((image_id, source_path, collection))
        return f"/managed/{image_id}.png"


def _image_encoder(monkeypatch, enabled=True):
    from src.ingestion.embedding import image_encoder as mod

    monkeypatch.setattr(mod.EmbeddingFactory, "create", staticmethod(lambda s: _FakeImageEmbedding()))
    settings = Settings()
    settings.ingestion.image_embedding = enabled
    return mod.ImageEncoder(settings, image_storage=_FakeStorage())


def test_image_encoder_disabled_returns_empty(monkeypatch):
    from src.core.types import Document

    enc = _image_encoder(monkeypatch, enabled=False)
    doc = Document(id="d1", text="t", metadata={"images": [{"id": "a", "path": "p"}]})
    assert enc.encode_document(doc) == []


def test_image_encoder_builds_image_records(monkeypatch):
    from src.core.types import Document

    enc = _image_encoder(monkeypatch, enabled=True)
    doc = Document(
        id="d1",
        text="t",
        metadata={
            "doc_hash": "h1",
            "images": [
                {"id": "h1_000", "path": "data/images/_staging/h1/h1_000.png", "page": 1},
            ],
        },
    )
    records = enc.encode_document(doc, collection="kb", captions={"h1_000": "a chart"})

    assert len(records) == 1
    rec = records[0]
    assert rec.id == "img_h1_000"
    assert rec.dense_vector == [1.0, 2.0, 3.0]
    assert rec.metadata["modality"] == "image"
    assert rec.metadata["image_refs"] == ["h1_000"]
    assert "a chart" in rec.text


# --------------------------------------------------------------------------- #
# DenseRetriever vector / image search
# --------------------------------------------------------------------------- #
class _FakeStore:
    def __init__(self):
        self.queried_vectors = []

    def query(self, vector, top_k=10, filters=None):
        self.queried_vectors.append(vector)
        return [SimpleNamespace(id="c1", score=0.9, text="hit", metadata={"modality": "image"})]


def test_dense_retrieve_by_vector():
    from src.core.query_engine.dense_retriever import DenseRetriever

    store = _FakeStore()
    retr = DenseRetriever(Settings(), embedding_client=_FakeImageEmbedding(), vector_store=store)
    results = retr.retrieve_by_vector([0.5, 0.5, 0.5], top_k=3)

    assert results[0].chunk_id == "c1"
    assert store.queried_vectors == [[0.5, 0.5, 0.5]]


def test_dense_embed_image_query():
    from src.core.query_engine.dense_retriever import DenseRetriever

    retr = DenseRetriever(Settings(), embedding_client=_FakeImageEmbedding(), vector_store=_FakeStore())
    assert retr.embed_image_query(b"bytes") == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# HybridSearch image routing
# --------------------------------------------------------------------------- #
def test_hybrid_image_only_is_dense_only():
    from src.core.query_engine.hybrid_search import HybridSearch
    from src.core.types import RetrievalResult

    class _Dense:
        def embed_image_query(self, image):
            return [9.0]

        def retrieve_by_vector(self, vector, top_k=10, filters=None):
            assert vector == [9.0]
            return [RetrievalResult(chunk_id="img_x", score=1.0, text="", metadata={})]

        def retrieve(self, *a, **k):  # must NOT be called for image-only
            raise AssertionError("text dense retrieval should be skipped")

    class _Sparse:
        def retrieve(self, *a, **k):
            raise AssertionError("sparse should be skipped for image-only")

    hs = HybridSearch(
        Settings(),
        dense_retriever=_Dense(),
        sparse_retriever=_Sparse(),
        vector_store=_FakeStore(),
    )
    results = hs.search(query="", image=b"img")
    assert [r.chunk_id for r in results] == ["img_x"]


# --------------------------------------------------------------------------- #
# QueryKnowledgeHubTool image argument
# --------------------------------------------------------------------------- #
def test_query_tool_requires_query_or_image():
    from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool
    from src.mcp_server.protocol_handler import JsonRpcError

    tool = QueryKnowledgeHubTool(Settings(), hybrid_search=SimpleNamespace())
    with pytest.raises(JsonRpcError):
        tool.run()


def test_query_tool_passes_validated_image(monkeypatch, tmp_path):
    from src.mcp_server.tools.query_knowledge_hub import QueryKnowledgeHubTool

    seen = {}

    class _Hybrid:
        def search(self, query="", top_k=None, filters=None, trace=None, image=None):
            seen["image"] = image
            seen["query"] = query
            return []

    tool = QueryKnowledgeHubTool(
        Settings(), hybrid_search=_Hybrid(), trace_collector=_DisabledCollector()
    )
    payload = base64.b64encode(b"imgdata").decode()
    tool.run(image=payload)
    assert seen["image"] == b"imgdata"


class _DisabledCollector:
    def collect(self, trace):
        pass
