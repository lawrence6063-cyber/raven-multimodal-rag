"""Tests for BaseLoader and PdfLoader contract."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.libs.loader.base_loader import BaseLoader, LoaderError
from src.libs.loader.pdf_loader import PdfLoader
from src.core.types import Document


class TestPdfLoader:
    """Test PdfLoader contract."""

    def test_file_not_found(self):
        loader = PdfLoader()
        with pytest.raises(LoaderError, match="not found"):
            loader.load("/nonexistent/file.pdf")

    def test_not_a_pdf(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("hello")
        loader = PdfLoader()
        with pytest.raises(LoaderError, match="Not a PDF"):
            loader.load(str(txt))

    @patch("markitdown.MarkItDown")
    def test_load_success(self, mock_markitdown_cls, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake content")

        mock_converter = MagicMock()
        mock_markitdown_cls.return_value = mock_converter
        mock_result = MagicMock()
        mock_result.text_content = "# Title\n\nSome content here."
        mock_converter.convert.return_value = mock_result

        loader = PdfLoader()
        doc = loader.load(str(pdf))

        assert isinstance(doc, Document)
        assert doc.text == "# Title\n\nSome content here."
        assert doc.metadata["source_path"] == str(pdf)
        assert doc.metadata["doc_type"] == "pdf"
        assert "doc_hash" in doc.metadata
        assert doc.id.startswith("doc_")

    @patch("markitdown.MarkItDown")
    def test_metadata_contains_required_fields(self, mock_cls, tmp_path):
        pdf = tmp_path / "sample.pdf"
        pdf.write_bytes(b"%PDF content")
        mock_cls.return_value.convert.return_value = MagicMock(text_content="text")

        loader = PdfLoader()
        doc = loader.load(str(pdf))

        assert "source_path" in doc.metadata
        assert "doc_type" in doc.metadata
        assert "file_name" in doc.metadata
        assert doc.metadata["file_name"] == "sample.pdf"

    @patch("markitdown.MarkItDown")
    def test_no_images_extracted_from_textonly_pdf(self, mock_cls, tmp_path):
        """A non-image PDF (and now-removed markdown regex) yields no phantom refs."""
        pdf = tmp_path / "with_img.pdf"
        pdf.write_bytes(b"%PDF content")
        text_with_img = "# Title\n\n![diagram](images/fig1.png)\n\nSome text."
        mock_cls.return_value.convert.return_value = MagicMock(text_content=text_with_img)

        loader = PdfLoader()
        doc = loader.load(str(pdf))

        # Markdown-regex extraction was replaced by real pypdfium2 extraction;
        # a fake/text-only PDF must not invent image references.
        assert doc.metadata["images"] == []

    @patch("markitdown.MarkItDown")
    def test_extracted_images_get_placeholders(self, mock_cls, tmp_path):
        """When images are extracted, [IMAGE: id] placeholders are appended to text."""
        pdf = tmp_path / "img.pdf"
        pdf.write_bytes(b"%PDF content")
        mock_cls.return_value.convert.return_value = MagicMock(text_content="Body text")

        loader = PdfLoader()
        fake_images = [
            {"id": "abc_000", "path": "data/images/_staging/abc/abc_000.png", "page": 0},
            {"id": "abc_001", "path": "data/images/_staging/abc/abc_001.png", "page": 1},
        ]
        with patch.object(PdfLoader, "_extract_images", return_value=fake_images):
            doc = loader.load(str(pdf))

        assert doc.metadata["images"] == fake_images
        assert "[IMAGE: abc_000]" in doc.text
        assert "[IMAGE: abc_001]" in doc.text

    def test_supported_extensions(self):
        loader = PdfLoader()
        assert ".pdf" in loader.supported_extensions
