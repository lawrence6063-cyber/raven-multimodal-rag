"""Configuration loading and validation.

Provides Settings dataclass and load/validate functions for config/settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class SettingsError(Exception):
    """Raised when configuration is invalid or missing required fields."""


@dataclass
class LLMSettings:
    """LLM provider configuration."""

    provider: str = ""
    model: str = ""
    api_key: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = "2024-02-01"
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass
class EmbeddingSettings:
    """Embedding provider configuration."""

    provider: str = ""
    model: str = ""
    api_key: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = "2024-02-01"
    base_url: str = ""
    dimensions: int = 1536


@dataclass
class VectorStoreSettings:
    """Vector store configuration."""

    provider: str = "chroma"
    collection_name: str = "default"
    persist_directory: str = "data/db/chroma"


@dataclass
class LoaderSettings:
    """Document loader configuration.

    provider 选择 PDF 解析后端：``markitdown``（默认，兼容旧行为）或
    ``pymupdf``（版面感知解析，修断词+双栏阅读顺序+表格抽取+图注定位）。
    """

    provider: str = "markitdown"
    image_output_dir: str = "data/images"
    # extract_tables 是否用 pdfplumber 抽取表格为 Markdown（仅 pymupdf provider 生效）
    extract_tables: bool = True
    # column_gap_ratio 双栏检测的列间距阈值（相对页宽），用于版面重排
    column_gap_ratio: float = 0.15


@dataclass
class SplitterSettings:
    """Splitter configuration."""

    provider: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200


@dataclass
class RetrievalSettings:
    """Retrieval configuration."""

    top_k: int = 10
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    rrf_k: int = 60


@dataclass
class RerankSettings:
    """Rerank configuration."""

    enabled: bool = False
    provider: str = "none"
    model: str = ""
    top_n: int = 5


@dataclass
class VisionLLMSettings:
    """Vision LLM configuration."""

    enabled: bool = False
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    azure_endpoint: str = ""
    deployment_name: str = ""
    api_version: str = "2024-02-01"
    base_url: str = ""
    max_image_size: int = 2048


@dataclass
class ChunkRefinerSettings:
    """Chunk refiner configuration."""

    use_llm: bool = False


@dataclass
class MetadataEnricherSettings:
    """Metadata enricher configuration."""

    use_llm: bool = False


@dataclass
class ImageCaptionerSettings:
    """Image captioner configuration."""

    enabled: bool = False


@dataclass
class IngestionSettings:
    """Ingestion pipeline configuration."""

    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(default_factory=MetadataEnricherSettings)
    image_captioner: ImageCaptionerSettings = field(default_factory=ImageCaptionerSettings)
    batch_size: int = 32
    bm25_index_path: str = "data/db/bm25"
    # image_embedding 是否将文档图片编码为独立多模态向量入库（需 embedding provider 支持图片）
    image_embedding: bool = False


@dataclass
class EvaluationSettings:
    """Evaluation configuration."""

    backends: list[str] = field(default_factory=lambda: ["custom"])
    golden_test_set: str = "tests/fixtures/golden_test_set.json"


@dataclass
class ObservabilitySettings:
    """Observability configuration."""

    trace_enabled: bool = True
    log_file: str = "logs/traces.jsonl"
    log_level: str = "INFO"


@dataclass
class Settings:
    """Root configuration object for the RAG MCP Server."""

    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    loader: LoaderSettings = field(default_factory=LoaderSettings)
    splitter: SplitterSettings = field(default_factory=SplitterSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    rerank: RerankSettings = field(default_factory=RerankSettings)
    vision_llm: VisionLLMSettings = field(default_factory=VisionLLMSettings)
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    observability: ObservabilitySettings = field(default_factory=ObservabilitySettings)


# Required fields that must have non-empty values
_REQUIRED_FIELDS = [
    "llm.provider",
    "llm.model",
    "embedding.provider",
    "embedding.model",
    "vector_store.provider",
]


def _get_nested(data: dict[str, Any], path: str) -> Any:
    """Get a nested value from a dict using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _build_dataclass(cls, data: dict[str, Any] | None):
    """Build a dataclass instance from a dict, ignoring unknown fields."""
    if data is None:
        return cls()
    known_fields = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return cls(**filtered)


def validate_settings(raw: dict[str, Any]) -> None:
    """Validate that all required fields exist and are non-empty.

    Raises:
        SettingsError: If any required field is missing or empty.
    """
    missing = []
    for field_path in _REQUIRED_FIELDS:
        value = _get_nested(raw, field_path)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            missing.append(field_path)

    if missing:
        raise SettingsError(
            f"Missing or empty required configuration fields: {', '.join(missing)}. "
            f"Please check config/settings.yaml"
        )


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    """Load and validate settings from a YAML file.

    Args:
        path: Path to the settings YAML file.

    Returns:
        Validated Settings object.

    Raises:
        SettingsError: If file not found, parse error, or validation failure.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise SettingsError(f"Configuration file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SettingsError(f"Failed to parse YAML configuration: {e}") from e

    if not isinstance(raw, dict):
        raise SettingsError("Configuration file must contain a YAML mapping (dict)")

    validate_settings(raw)

    # Build nested settings
    ingestion_raw = raw.get("ingestion", {}) or {}
    ingestion_settings = IngestionSettings(
        chunk_refiner=_build_dataclass(ChunkRefinerSettings, ingestion_raw.get("chunk_refiner")),
        metadata_enricher=_build_dataclass(MetadataEnricherSettings, ingestion_raw.get("metadata_enricher")),
        image_captioner=_build_dataclass(ImageCaptionerSettings, ingestion_raw.get("image_captioner")),
        batch_size=ingestion_raw.get("batch_size", 32),
        bm25_index_path=ingestion_raw.get("bm25_index_path", "data/db/bm25"),
        image_embedding=ingestion_raw.get("image_embedding", False),
    )

    return Settings(
        llm=_build_dataclass(LLMSettings, raw.get("llm")),
        embedding=_build_dataclass(EmbeddingSettings, raw.get("embedding")),
        vector_store=_build_dataclass(VectorStoreSettings, raw.get("vector_store")),
        loader=_build_dataclass(LoaderSettings, raw.get("loader")),
        splitter=_build_dataclass(SplitterSettings, raw.get("splitter")),
        retrieval=_build_dataclass(RetrievalSettings, raw.get("retrieval")),
        rerank=_build_dataclass(RerankSettings, raw.get("rerank")),
        vision_llm=_build_dataclass(VisionLLMSettings, raw.get("vision_llm")),
        ingestion=ingestion_settings,
        evaluation=_build_dataclass(EvaluationSettings, raw.get("evaluation")),
        observability=_build_dataclass(ObservabilitySettings, raw.get("observability")),
    )
