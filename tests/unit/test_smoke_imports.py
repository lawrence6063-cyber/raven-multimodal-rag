"""Smoke tests — verify all key packages can be imported."""

import pytest


class TestSmokeImports:
    """Verify that all top-level and sub-packages are importable."""

    def test_import_top_level_packages(self):
        """All top-level source packages should be importable."""
        from src import mcp_server, core, ingestion, libs, observability
        assert mcp_server is not None
        assert core is not None
        assert ingestion is not None
        assert libs is not None
        assert observability is not None

    def test_import_mcp_server_subpackages(self):
        """MCP server sub-packages should be importable."""
        from src.mcp_server import tools
        assert tools is not None

    def test_import_core_subpackages(self):
        """Core sub-packages should be importable."""
        from src.core import query_engine, response, trace
        assert query_engine is not None
        assert response is not None
        assert trace is not None

    def test_import_ingestion_subpackages(self):
        """Ingestion sub-packages should be importable."""
        from src.ingestion import chunking, transform, embedding, storage
        assert chunking is not None
        assert transform is not None
        assert embedding is not None
        assert storage is not None

    def test_import_libs_subpackages(self):
        """Libs sub-packages should be importable."""
        from src.libs import loader, llm, embedding, splitter, vector_store, reranker, evaluator
        assert loader is not None
        assert llm is not None
        assert embedding is not None
        assert splitter is not None
        assert vector_store is not None
        assert reranker is not None
        assert evaluator is not None

    def test_import_observability_subpackages(self):
        """Observability sub-packages should be importable."""
        from src.observability import dashboard, evaluation, logger
        assert dashboard is not None
        assert evaluation is not None
        assert logger is not None

    def test_import_dashboard_subpackages(self):
        """Dashboard sub-packages should be importable."""
        from src.observability.dashboard import pages, services
        assert pages is not None
        assert services is not None

    def test_logger_get_logger(self):
        """Logger module should provide get_logger function."""
        from src.observability.logger import get_logger
        log = get_logger("test")
        assert log is not None
        assert log.name == "test"

    def test_config_files_exist(self):
        """Configuration files should exist and be readable."""
        from pathlib import Path

        config_files = [
            "config/settings.yaml",
            "config/prompts/image_captioning.txt",
            "config/prompts/chunk_refinement.txt",
            "config/prompts/rerank.txt",
        ]
        for f in config_files:
            path = Path(f)
            assert path.exists(), f"Config file missing: {f}"
            assert path.read_text(encoding="utf-8"), f"Config file empty: {f}"

    def test_main_entry_point_exists(self):
        """main.py should be importable."""
        import main
        assert hasattr(main, "main")
