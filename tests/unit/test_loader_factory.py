"""Tests for LoaderFactory routing and built-in provider registration."""

import pytest

from src.core.settings import LoaderSettings
from src.libs.loader.base_loader import BaseLoader, LoaderError
from src.libs.loader.loader_factory import (
    LoaderFactory,
    register_loader,
    _LOADER_REGISTRY,
)
from src.libs.loader.pdf_loader import PdfLoader
from src.libs.loader.pymupdf_loader import PyMuPDFLoader


class FakeLoader(BaseLoader):
    """Fake loader for registry isolation tests."""

    def __init__(self, settings=None):
        self.settings = settings

    def load(self, path):  # pragma: no cover - not exercised
        raise NotImplementedError

    @property
    def supported_extensions(self):
        return [".pdf"]


class TestLoaderFactory:
    """Test LoaderFactory routing logic."""

    def setup_method(self):
        self._original = _LOADER_REGISTRY.copy()

    def teardown_method(self):
        _LOADER_REGISTRY.clear()
        _LOADER_REGISTRY.update(self._original)

    def test_builtin_providers_available(self):
        providers = LoaderFactory.available_providers()
        assert "markitdown" in providers
        assert "pymupdf" in providers

    def test_create_markitdown(self):
        loader = LoaderFactory.create(LoaderSettings(provider="markitdown"))
        assert isinstance(loader, PdfLoader)

    def test_create_pymupdf(self):
        loader = LoaderFactory.create(LoaderSettings(provider="pymupdf"))
        assert isinstance(loader, PyMuPDFLoader)

    def test_provider_is_case_insensitive(self):
        loader = LoaderFactory.create(LoaderSettings(provider="PyMuPDF"))
        assert isinstance(loader, PyMuPDFLoader)

    def test_unknown_provider_raises(self):
        with pytest.raises(LoaderError, match="Unknown loader provider"):
            LoaderFactory.create(LoaderSettings(provider="nonexistent"))

    def test_register_decorator(self):
        register_loader("fake")(FakeLoader)
        loader = LoaderFactory.create(LoaderSettings(provider="fake"))
        assert isinstance(loader, FakeLoader)

    def test_image_output_dir_propagates_to_loader(self):
        loader = LoaderFactory.create(
            LoaderSettings(provider="pymupdf", image_output_dir="data/custom_imgs")
        )
        assert str(loader._image_output_dir) == "data/custom_imgs"
