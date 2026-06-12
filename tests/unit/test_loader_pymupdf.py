"""Tests for PyMuPDFLoader layout-aware parsing helpers.

External libraries (``fitz`` / ``pdfplumber``) are not required here: the pure
layout/table/caption helpers operate on plain Python structures, and page objects
are mocked. This keeps the suite offline and fast.
"""

import pytest
from unittest.mock import MagicMock

from src.core.settings import LoaderSettings
from src.libs.loader.base_loader import LoaderError
from src.libs.loader.pymupdf_loader import PyMuPDFLoader


def _loader():
    return PyMuPDFLoader(LoaderSettings(provider="pymupdf"))


class TestValidation:
    def test_file_not_found(self):
        with pytest.raises(LoaderError, match="not found"):
            _loader().load("/nonexistent/file.pdf")

    def test_not_a_pdf(self, tmp_path):
        txt = tmp_path / "doc.txt"
        txt.write_text("hello")
        with pytest.raises(LoaderError, match="Not a PDF"):
            _loader().load(str(txt))

    def test_supported_extensions(self):
        assert ".pdf" in _loader().supported_extensions


class TestDehyphenation:
    def test_merge_trailing_hyphen(self):
        text = PyMuPDFLoader._join_lines_dehyphen(["we esti-", "mate the fraction"])
        assert text == "we estimate the fraction"

    def test_join_lines_with_space(self):
        text = PyMuPDFLoader._join_lines_dehyphen(["around 96% of", "our dataset"])
        assert text == "around 96% of our dataset"

    def test_skip_blank_lines(self):
        assert PyMuPDFLoader._join_lines_dehyphen(["", "hello", "", "world"]) == "hello world"


class TestTextElements:
    def test_words_become_space_separated_block(self):
        # Each word is a separate token → spaces are never lost (fixes "around96%").
        words = [
            (0, 0, 10, 8, "around", 0, 0, 0),
            (12, 0, 20, 8, "96%", 0, 0, 1),
            (22, 0, 30, 8, "of", 0, 0, 2),
            (0, 10, 10, 18, "our", 0, 1, 0),
            (12, 10, 25, 18, "dataset", 0, 1, 1),
        ]
        page = MagicMock()
        page.get_text.return_value = words
        elems = _loader()._extract_text_elements(page)
        assert len(elems) == 1
        assert elems[0]["text"] == "around 96% of our dataset"
        assert elems[0]["type"] == "text"

    def test_empty_words_returns_empty(self):
        page = MagicMock()
        page.get_text.return_value = []
        assert _loader()._extract_text_elements(page) == []


class TestReadingOrder:
    def _el(self, x0, y0, x1, y1, text):
        return {"bbox": (x0, y0, x1, y1), "type": "text", "text": text}

    def test_two_column_left_then_right(self):
        # page width 200, mid=100. Left column x<100, right column x>=100.
        elems = [
            self._el(110, 10, 190, 20, "R1"),
            self._el(10, 10, 90, 20, "L1"),
            self._el(10, 30, 90, 40, "L2"),
            self._el(110, 30, 190, 40, "R2"),
        ]
        ordered = _loader()._order_elements(elems, page_width=200)
        assert [e["text"] for e in ordered] == ["L1", "L2", "R1", "R2"]

    def test_single_column_top_to_bottom(self):
        elems = [
            self._el(10, 50, 190, 60, "B"),
            self._el(10, 10, 190, 20, "A"),
        ]
        ordered = _loader()._order_elements(elems, page_width=200)
        assert [e["text"] for e in ordered] == ["A", "B"]

    def test_is_two_column_detection(self):
        loader = _loader()
        two = [
            {"bbox": (10, 10, 90, 20)},
            {"bbox": (10, 30, 90, 40)},
            {"bbox": (110, 10, 190, 20)},
            {"bbox": (110, 30, 190, 40)},
        ]
        assert loader._is_two_column(two, mid=100) is True
        one = [{"bbox": (10, 10, 190, 20)}, {"bbox": (10, 30, 190, 40)}]
        assert loader._is_two_column(one, mid=100) is False


class TestTableMarkdown:
    def test_basic_table(self):
        md = PyMuPDFLoader._table_to_markdown([["Model", "BLEU"], ["A", "26.3"], ["B", "27.1"]])
        lines = md.splitlines()
        assert lines[0] == "| Model | BLEU |"
        assert lines[1] == "| --- | --- |"
        assert "| A | 26.3 |" in lines
        assert "| B | 27.1 |" in lines

    def test_none_cells_and_ragged_rows(self):
        md = PyMuPDFLoader._table_to_markdown([["a", None], ["b"]])
        assert md.splitlines()[0] == "| a |  |"
        # ragged row padded to width 2
        assert "| b |  |" in md

    def test_pipe_escaped(self):
        md = PyMuPDFLoader._table_to_markdown([["x|y"], ["z"]])
        assert "x\\|y" in md

    def test_empty_returns_empty(self):
        assert PyMuPDFLoader._table_to_markdown([]) == ""
        assert PyMuPDFLoader._table_to_markdown([[None, None]]) == ""


class TestDropTextInsideTables:
    def test_text_inside_table_dropped(self):
        text_elems = [
            {"bbox": (10, 10, 90, 20), "type": "text", "text": "outside"},
            {"bbox": (15, 55, 80, 65), "type": "text", "text": "inside-table"},
        ]
        table_elems = [{"bbox": (10, 50, 100, 100), "type": "table", "text": "| a |"}]
        kept = PyMuPDFLoader._drop_text_inside_tables(text_elems, table_elems)
        assert [e["text"] for e in kept] == ["outside"]

    def test_no_tables_keeps_all(self):
        text_elems = [{"bbox": (10, 10, 90, 20), "type": "text", "text": "a"}]
        assert PyMuPDFLoader._drop_text_inside_tables(text_elems, []) == text_elems


class TestFindCaption:
    def test_finds_figure_caption_below_image(self):
        page = MagicMock()
        page.get_text.return_value = [
            (10, 5, 100, 50, "some body text", 0, 0),       # above image, no figure
            (10, 105, 100, 120, "Figure 1: The architecture", 1, 0),  # below image
        ]
        # image bbox bottom at y=100
        cap = PyMuPDFLoader._find_caption(page, (10, 60, 100, 100))
        assert cap == "Figure 1: The architecture"

    def test_no_caption_returns_empty(self):
        page = MagicMock()
        page.get_text.return_value = [(10, 105, 100, 120, "just a paragraph", 0, 0)]
        assert PyMuPDFLoader._find_caption(page, (10, 60, 100, 100)) == ""

    def test_none_bbox_returns_empty(self):
        page = MagicMock()
        assert PyMuPDFLoader._find_caption(page, None) == ""


class TestFallback:
    def test_fallback_uses_markitdown_loader(self, tmp_path, monkeypatch):
        from unittest.mock import patch

        pdf = tmp_path / "f.pdf"
        pdf.write_bytes(b"%PDF content")

        # Force the fitz import inside load() to fail → fallback path.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "fitz":
                raise ImportError("no fitz")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with patch("markitdown.MarkItDown") as mock_cls:
            mock_cls.return_value.convert.return_value = MagicMock(text_content="fallback text")
            doc = _loader().load(str(pdf))

        assert doc.text == "fallback text"
        assert doc.metadata["doc_type"] == "pdf"
