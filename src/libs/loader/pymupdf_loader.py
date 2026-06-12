"""PyMuPDFLoader — layout-aware PDF parsing (OPTIMIZATION_SPEC §2 P0).

Why this loader exists
----------------------
MarkItDown (the legacy ``markitdown`` provider) often drops spaces between glyph
runs in two-column academic PDFs (e.g. ``around96%ofourdataset``) and emits the
two columns interleaved in the wrong reading order. This loader uses ``PyMuPDF``
(``fitz``) coordinate information to:

1. **Reading order (2.1)** — group text blocks into columns by their x position
   and emit them top-to-bottom, left-column-first, fixing two-column ordering.
2. **De-hyphenation / spacing (2.1)** — rebuild lines from word-level tokens (each
   carries its own bbox) so word boundaries are never lost, and merge trailing
   ``-`` line breaks.
3. **Table extraction (2.2)** — detect tables with ``pdfplumber`` and render them
   as Markdown tables anchored at their on-page position; text inside the table
   region is suppressed to avoid duplication.
4. **Figure anchoring (2.3)** — place each ``[IMAGE: <id>]`` placeholder at the
   image's real position (not appended at the end) and attach the nearby
   ``Figure N: ...`` caption to the image descriptor.

Resilience: if ``fitz`` is unavailable the loader falls back to the markitdown
provider. ``pdfplumber``/per-page/per-image failures are logged and skipped so a
single bad page never aborts loading. Output stays compatible with the
``Document(text, metadata)`` contract consumed downstream.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader, LoaderError
from src.libs.loader.loader_factory import register_loader
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import LoaderSettings

logger = get_logger("loader.pymupdf")

# _MIN_IMAGE_DIM smallest width/height (px) kept; filters icons/glyph fragments
_MIN_IMAGE_DIM = 96

# _FIGURE_RE 匹配图注前缀，如 "Figure 1", "Fig. 2", "图 3"
_FIGURE_RE = re.compile(r"(?:fig(?:ure)?\.?\s*\d+|图\s*\d+)", re.IGNORECASE)


@register_loader("pymupdf")
class PyMuPDFLoader(BaseLoader):
    """Layout-aware PDF loader (text reorder + table extraction + figure anchoring)."""

    def __init__(self, settings: "LoaderSettings | None" = None):
        image_output_dir = "data/images"
        extract_tables = True
        column_gap_ratio = 0.15
        if settings is not None:
            image_output_dir = getattr(settings, "image_output_dir", image_output_dir)
            extract_tables = getattr(settings, "extract_tables", extract_tables)
            column_gap_ratio = getattr(settings, "column_gap_ratio", column_gap_ratio)
        self._settings = settings
        self._image_output_dir = Path(image_output_dir)
        self._extract_tables = bool(extract_tables)
        self._column_gap_ratio = float(column_gap_ratio)

    def load(self, path: str) -> Document:
        """Load a PDF with layout-aware parsing; fall back to markitdown on failure."""
        file_path = Path(path)
        if not file_path.exists():
            raise LoaderError(f"File not found: {path}", path=path)
        if file_path.suffix.lower() != ".pdf":
            raise LoaderError(f"Not a PDF file: {path}", path=path)

        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.warning("PyMuPDF (fitz) not installed; falling back to markitdown loader")
            return self._fallback_load(path)

        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
        doc_id = f"doc_{doc_hash}"

        try:
            tables_by_page = self._extract_tables_by_page(file_path) if self._extract_tables else {}
            text, images = self._parse_document(fitz, file_path, doc_hash, tables_by_page)
        except Exception as e:
            logger.warning(f"PyMuPDF parsing failed for {file_path.name}: {e}; falling back to markitdown")
            return self._fallback_load(path)

        metadata: dict[str, Any] = {
            "source_path": str(file_path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "file_name": file_path.name,
            "images": images,
        }
        return Document(id=doc_id, text=text, metadata=metadata)

    def _fallback_load(self, path: str) -> Document:
        """Delegate to the markitdown PdfLoader (lazy import avoids cycles)."""
        from src.libs.loader.pdf_loader import PdfLoader

        return PdfLoader(self._settings).load(path)

    # ------------------------------------------------------------------ parse
    def _parse_document(
        self, fitz, file_path: Path, doc_hash: str, tables_by_page: dict[int, list[dict]]
    ) -> tuple[str, list[dict]]:
        """Render the whole PDF into reading-ordered Markdown + image descriptors."""
        staging_dir = self._image_output_dir / "_staging" / doc_hash
        staging_dir.mkdir(parents=True, exist_ok=True)

        page_texts: list[str] = []
        images: list[dict] = []
        img_index = 0

        pdf = fitz.open(str(file_path))
        try:
            for page_num in range(pdf.page_count):
                try:
                    page = pdf[page_num]
                    text_elems = self._extract_text_elements(page)
                    table_elems = tables_by_page.get(page_num, [])
                    text_elems = self._drop_text_inside_tables(text_elems, table_elems)
                    image_elems, page_images = self._extract_image_elements(
                        fitz, pdf, page, page_num, doc_hash, staging_dir, img_index
                    )
                    img_index += len(page_images)
                    images.extend(page_images)

                    elements = text_elems + table_elems + image_elems
                    page_width = float(getattr(page.rect, "width", 0.0) or 0.0)
                    ordered = self._order_elements(elements, page_width)
                    page_text = "\n\n".join(e["text"] for e in ordered if e.get("text"))
                    if page_text.strip():
                        page_texts.append(page_text)
                except Exception as e:  # per-page resilience
                    logger.warning(f"PyMuPDF failed on page {page_num}: {e}")
        finally:
            pdf.close()

        if images:
            logger.info(f"Extracted {len(images)} image(s) from {file_path.name}")
        return "\n\n".join(page_texts), images

    # ------------------------------------------------------------- text (2.1)
    def _extract_text_elements(self, page) -> list[dict]:
        """Build per-block text elements from word tokens (space-preserving)."""
        try:
            words = page.get_text("words")  # (x0,y0,x1,y1,word,block_no,line_no,word_no)
        except Exception as e:
            logger.warning(f"get_text('words') failed: {e}")
            return []
        if not words:
            return []

        # block_no -> line_no -> list[(word_no, x0, y0, x1, y1, word)]
        blocks: dict[int, dict[int, list[tuple]]] = {}
        for w in words:
            if len(w) < 8:
                continue
            x0, y0, x1, y1, word, bno, lno, wno = w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]
            if not str(word).strip():
                continue
            blocks.setdefault(bno, {}).setdefault(lno, []).append((wno, x0, y0, x1, y1, word))

        elements: list[dict] = []
        for bno in sorted(blocks):
            lines = blocks[bno]
            line_texts: list[str] = []
            xs0: list[float] = []
            ys0: list[float] = []
            xs1: list[float] = []
            ys1: list[float] = []
            for lno in sorted(lines):
                tokens = sorted(lines[lno], key=lambda t: t[0])
                line_texts.append(" ".join(str(t[5]) for t in tokens))
                for t in tokens:
                    xs0.append(t[1])
                    ys0.append(t[2])
                    xs1.append(t[3])
                    ys1.append(t[4])
            block_text = self._join_lines_dehyphen(line_texts)
            if not block_text.strip():
                continue
            bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
            elements.append({"bbox": bbox, "type": "text", "text": block_text})
        return elements

    @staticmethod
    def _join_lines_dehyphen(lines: list[str]) -> str:
        """Join wrapped lines into a paragraph, merging trailing-hyphen breaks."""
        out = ""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if not out:
                out = line
            elif out.endswith("-"):
                out = out[:-1] + line
            else:
                out = f"{out} {line}"
        return out

    # ----------------------------------------------------------- tables (2.2)
    def _extract_tables_by_page(self, file_path: Path) -> dict[int, list[dict]]:
        """Detect tables with pdfplumber → ``{page_index: [table_element, ...]}``."""
        try:
            import pdfplumber
        except ImportError:
            logger.warning("pdfplumber not installed; skipping table extraction")
            return {}

        result: dict[int, list[dict]] = {}
        try:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    try:
                        tables = page.find_tables()
                    except Exception as e:  # per-page resilience
                        logger.warning(f"Table detection failed on page {page_index}: {e}")
                        continue
                    elems: list[dict] = []
                    for tbl in tables or []:
                        try:
                            data = tbl.extract()
                            md = self._table_to_markdown(data)
                            if not md:
                                continue
                            elems.append({"bbox": tuple(tbl.bbox), "type": "table", "text": md})
                        except Exception as e:  # per-table resilience
                            logger.warning(f"Table extraction failed on page {page_index}: {e}")
                    if elems:
                        result[page_index] = elems
        except Exception as e:
            logger.warning(f"pdfplumber failed for {file_path.name}: {e}")
            return {}
        return result

    @staticmethod
    def _table_to_markdown(data: list[list]) -> str:
        """Render a 2D cell matrix as a GitHub-flavored Markdown table."""
        rows = [
            [("" if cell is None else str(cell).replace("\n", " ").replace("|", "\\|").strip()) for cell in row]
            for row in (data or [])
            if row is not None
        ]
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        header = rows[0]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
        ]
        for r in rows[1:]:
            lines.append("| " + " | ".join(r) + " |")
        return "\n".join(lines)

    @staticmethod
    def _drop_text_inside_tables(text_elems: list[dict], table_elems: list[dict]) -> list[dict]:
        """Suppress text blocks whose center lies inside any detected table bbox."""
        if not table_elems:
            return text_elems
        boxes = [t["bbox"] for t in table_elems]
        kept: list[dict] = []
        for e in text_elems:
            x0, y0, x1, y1 = e["bbox"]
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            inside = any(bx0 <= cx <= bx1 and by0 <= cy <= by1 for bx0, by0, bx1, by1 in boxes)
            if not inside:
                kept.append(e)
        return kept

    # ----------------------------------------------------------- images (2.3)
    def _extract_image_elements(
        self, fitz, pdf, page, page_num: int, doc_hash: str, staging_dir: Path, index_start: int
    ) -> tuple[list[dict], list[dict]]:
        """Save embedded images, anchor placeholders by bbox, attach figure captions."""
        try:
            raw_images = page.get_images(full=True)
        except Exception as e:
            logger.warning(f"get_images failed on page {page_num}: {e}")
            return [], []

        elements: list[dict] = []
        images: list[dict] = []
        index = index_start
        for img in raw_images or []:
            try:
                xref = img[0]
                bbox = self._image_bbox(page, xref)
                saved = self._save_image(fitz, pdf, xref, staging_dir, doc_hash, index, page_num)
                if saved is None:
                    continue
                caption = self._find_caption(page, bbox)
                if bbox is not None:
                    saved["bbox"] = [float(v) for v in bbox]
                if caption:
                    saved["caption"] = caption
                images.append(saved)
                element_bbox = bbox if bbox is not None else (0.0, 1e9, 0.0, 1e9)
                elements.append(
                    {"bbox": tuple(element_bbox), "type": "image", "text": f"[IMAGE: {saved['id']}]"}
                )
                index += 1
            except Exception as e:  # per-image resilience
                logger.warning(f"Image handling failed on page {page_num}: {e}")
        return elements, images

    @staticmethod
    def _image_bbox(page, xref):
        """Return the first on-page rect for an image xref, or None."""
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            return None
        if not rects:
            return None
        r = rects[0]
        return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))

    def _save_image(
        self, fitz, pdf, xref: int, staging_dir: Path, doc_hash: str, index: int, page_num: int
    ) -> dict | None:
        """Render an image xref to PNG; return its descriptor or None when filtered."""
        try:
            pix = fitz.Pixmap(pdf, xref)
            if pix.width < _MIN_IMAGE_DIM or pix.height < _MIN_IMAGE_DIM:
                return None
            # Normalize CMYK / alpha to RGB so PNG save is always valid.
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_id = f"{doc_hash}_{index:03d}"
            dest = staging_dir / f"{image_id}.png"
            pix.save(str(dest))
            return {"id": image_id, "path": str(dest), "page": page_num}
        except Exception as e:  # per-image resilience
            logger.warning(f"Failed to save embedded image #{index}: {e}")
            return None

    @staticmethod
    def _find_caption(page, bbox) -> str:
        """Find the nearest ``Figure N: ...`` caption below/overlapping the image."""
        if bbox is None:
            return ""
        try:
            blocks = page.get_text("blocks")  # (x0,y0,x1,y1,text,block_no,block_type)
        except Exception:
            return ""
        bx0, by0, bx1, by1 = bbox
        candidates: list[tuple[float, str]] = []
        for blk in blocks or []:
            if len(blk) < 5:
                continue
            x0, y0, x1, y1, text = blk[0], blk[1], blk[2], blk[3], blk[4]
            text = " ".join(str(text).split())
            if not text or not _FIGURE_RE.search(text):
                continue
            horizontally_overlaps = x0 < bx1 and x1 > bx0
            if y0 >= by1 - 5 and horizontally_overlaps:
                candidates.append((abs(y0 - by1), text))
        if not candidates:
            return ""
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    # ------------------------------------------------------------- ordering
    def _order_elements(self, elements: list[dict], page_width: float) -> list[dict]:
        """Order elements in reading order, handling two-column layouts."""
        if not elements:
            return []
        if page_width <= 0:
            return sorted(elements, key=lambda e: (e["bbox"][1], e["bbox"][0]))
        mid = page_width / 2.0
        if self._is_two_column(elements, mid):
            return sorted(
                elements,
                key=lambda e: (self._column_index(e["bbox"], mid), e["bbox"][1], e["bbox"][0]),
            )
        return sorted(elements, key=lambda e: (e["bbox"][1], e["bbox"][0]))

    @staticmethod
    def _column_index(bbox, mid: float) -> int:
        """Right column when the element starts at/after the page midline."""
        return 1 if bbox[0] >= mid else 0

    @staticmethod
    def _is_two_column(elements: list[dict], mid: float) -> bool:
        """Heuristic: enough blocks fully left and fully right of the midline."""
        left = sum(1 for e in elements if e["bbox"][2] <= mid)
        right = sum(1 for e in elements if e["bbox"][0] >= mid)
        return left >= 2 and right >= 2

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]
