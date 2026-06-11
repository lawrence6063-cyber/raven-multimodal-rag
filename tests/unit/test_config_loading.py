"""Tests for configuration loading and validation."""

import pytest
import tempfile
from pathlib import Path

from src.core.settings import (
    Settings,
    SettingsError,
    load_settings,
    validate_settings,
)


VALID_CONFIG = """
llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "sk-test"
  temperature: 0.0
  max_tokens: 4096

embedding:
  provider: "openai"
  model: "text-embedding-ada-002"
  api_key: "sk-test"
  dimensions: 1536

vector_store:
  provider: "chroma"
  collection_name: "default"
  persist_directory: "data/db/chroma"

splitter:
  provider: "recursive"
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  top_k: 10
  dense_weight: 0.7
  sparse_weight: 0.3

rerank:
  enabled: false
  provider: "none"

vision_llm:
  enabled: false

ingestion:
  chunk_refiner:
    use_llm: false
  batch_size: 32

evaluation:
  backends: ["custom"]

observability:
  trace_enabled: true
  log_level: "INFO"
"""


class TestLoadSettings:
    """Test load_settings function."""

    def test_load_valid_config(self, tmp_path):
        """Should load a valid config file successfully."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(VALID_CONFIG)

        settings = load_settings(config_file)

        assert isinstance(settings, Settings)
        assert settings.llm.provider == "openai"
        assert settings.llm.model == "gpt-4o"
        assert settings.embedding.provider == "openai"
        assert settings.embedding.dimensions == 1536
        assert settings.vector_store.provider == "chroma"
        assert settings.splitter.chunk_size == 1000

    def test_file_not_found(self):
        """Should raise SettingsError if config file does not exist."""
        with pytest.raises(SettingsError, match="not found"):
            load_settings("/nonexistent/path/settings.yaml")

    def test_invalid_yaml(self, tmp_path):
        """Should raise SettingsError for malformed YAML."""
        config_file = tmp_path / "bad.yaml"
        config_file.write_text("  invalid:\n    - [bad yaml\n")

        with pytest.raises(SettingsError, match="Failed to parse"):
            load_settings(config_file)

    def test_non_dict_yaml(self, tmp_path):
        """Should raise SettingsError if YAML root is not a dict."""
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2\n")

        with pytest.raises(SettingsError, match="YAML mapping"):
            load_settings(config_file)

    def test_load_project_config(self):
        """Should load the actual project config/settings.yaml."""
        settings = load_settings("config/settings.yaml")
        assert settings.llm.provider == "qwen"
        assert settings.embedding.provider == "qwen_multimodal"
        assert settings.embedding.model == "multimodal-embedding-v1"


class TestValidateSettings:
    """Test validate_settings function."""

    def test_missing_llm_provider(self, tmp_path):
        """Should raise SettingsError when llm.provider is empty."""
        config = """
llm:
  provider: ""
  model: "gpt-4o"
embedding:
  provider: "openai"
  model: "text-embedding-ada-002"
vector_store:
  provider: "chroma"
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config)

        with pytest.raises(SettingsError, match="llm.provider"):
            load_settings(config_file)

    def test_missing_embedding_provider(self, tmp_path):
        """Should raise SettingsError when embedding.provider is missing."""
        config = """
llm:
  provider: "openai"
  model: "gpt-4o"
embedding:
  provider: ""
  model: "ada"
vector_store:
  provider: "chroma"
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config)

        with pytest.raises(SettingsError, match="embedding.provider"):
            load_settings(config_file)

    def test_missing_multiple_fields(self, tmp_path):
        """Should report all missing fields in error message."""
        config = """
llm:
  provider: ""
  model: ""
embedding:
  provider: ""
  model: ""
vector_store:
  provider: ""
"""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(config)

        with pytest.raises(SettingsError) as exc_info:
            load_settings(config_file)

        error_msg = str(exc_info.value)
        assert "llm.provider" in error_msg
        assert "llm.model" in error_msg
        assert "embedding.provider" in error_msg

    def test_valid_settings_pass(self):
        """Should not raise for valid settings."""
        raw = {
            "llm": {"provider": "openai", "model": "gpt-4o"},
            "embedding": {"provider": "openai", "model": "ada"},
            "vector_store": {"provider": "chroma"},
        }
        validate_settings(raw)  # Should not raise


class TestSettingsDataclass:
    """Test Settings dataclass defaults."""

    def test_default_settings(self):
        """Default Settings should have sensible defaults."""
        s = Settings()
        assert s.llm.temperature == 0.0
        assert s.embedding.dimensions == 1536
        assert s.vector_store.provider == "chroma"
        assert s.splitter.provider == "recursive"
        assert s.rerank.enabled is False
        assert s.ingestion.batch_size == 32

    def test_nested_ingestion_settings(self, tmp_path):
        """Should correctly parse nested ingestion settings."""
        config_file = tmp_path / "settings.yaml"
        config_file.write_text(VALID_CONFIG)

        settings = load_settings(config_file)

        assert settings.ingestion.chunk_refiner.use_llm is False
        assert settings.ingestion.batch_size == 32
