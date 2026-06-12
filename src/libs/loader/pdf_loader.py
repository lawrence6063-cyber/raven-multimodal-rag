"""PDF Loader — converts PDF to Markdown text and extracts embedded images.

Text extraction uses MarkItDown (good Markdown structure). Embedded raster
images are pulled out separately with ``pypdfium2`` (MarkItDown only yields plain
text), saved to a staging directory, and recorded in ``Document.metadata['images']``.
Per-image ``[IMAGE: <id>]`` placeholders are appended to the text so the chunker
can attach captions, and a downstream encoder can embed each image into the shared
multimodal vector space for cross-modal retrieval.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.types import Document
from src.libs.loader.base_loader import BaseLoader, LoaderError
from src.libs.loader.loader_factory import register_loader
from src.observability.logger import get_logger

if TYPE_CHECKING:
    from src.core.settings import LoaderSettings

logger = get_logger("loader.pdf")

# _MIN_IMAGE_DIM smallest width/height (px) kept; filters icons/glyph fragments
_MIN_IMAGE_DIM = 96


@register_loader("markitdown")
class PdfLoader(BaseLoader):
    """PDF loader: MarkItDown text + pypdfium2 embedded-image extraction."""

    def __init__(self, settings: "LoaderSettings | None" = None, image_output_dir: str = "data/images"):
        if settings is not None and getattr(settings, "image_output_dir", None):
            image_output_dir = settings.image_output_dir
        self._image_output_dir = Path(image_output_dir)

    def load(self, path: str) -> Document:
        """Load a PDF and convert to a Markdown Document with extracted images."""
        file_path = Path(path)
        if not file_path.exists():
            raise LoaderError(f"File not found: {path}", path=path)
        if file_path.suffix.lower() != ".pdf":
            raise LoaderError(f"Not a PDF file: {path}", path=path)

        try:
            from markitdown import MarkItDown

            converter = MarkItDown()
            result = converter.convert(str(file_path))
            text = result.text_content if result.text_content else ""
        except ImportError:
            raise LoaderError(
                "markitdown not installed. Run: pip install markitdown",
                path=path,
            )
        except Exception as e:
            raise LoaderError(f"PDF parsing failed: {e}", path=path) from e

        doc_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
        doc_id = f"doc_{doc_hash}"

        images = self._extract_images(file_path, doc_hash)
        if images:
            text = self._append_image_placeholders(text, images)

        metadata: dict = {
            "source_path": str(file_path),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
            "file_name": file_path.name,
            "images": images,
        }

        return Document(id=doc_id, text=text, metadata=metadata)

    def _extract_images(self, file_path: Path, doc_hash: str) -> list[dict]:
        """Extract embedded raster images from the PDF using pypdfium2.

        Returns a list of ``{id, path, page}`` dicts. Each image is saved as a PNG
        under ``<image_output_dir>/_staging/<doc_hash>/``. Failures on individual
        images/pages are logged and skipped so one bad image never aborts loading.
        """
        try:
            import pypdfium2 as pdfium
        except ImportError:
            logger.warning("pypdfium2 not installed; skipping image extraction")
            return []

        staging_dir = self._image_output_dir / "_staging" / doc_hash
        staging_dir.mkdir(parents=True, exist_ok=True)

        images: list[dict] = []
        pdf = None
        try:
            pdf = pdfium.PdfDocument(str(file_path))
            index = 0
            for page_num in range(len(pdf)):
                try:
                    page = pdf[page_num]
                    for obj in page.get_objects():
                        if getattr(obj, "type", None) != pdfium.raw.FPDF_PAGEOBJ_IMAGE:
                            continue
                        saved = self._save_image_object(obj, staging_dir, doc_hash, index, page_num)
                        if saved is not None:
                            images.append(saved)
                            index += 1
                except Exception as e:  # pragma: no cover - per-page resilience
                    logger.warning(f"Image extraction failed on page {page_num}: {e}")
        except Exception as e:
            logger.warning(f"PDF image extraction failed for {file_path.name}: {e}")
        finally:
            if pdf is not None:
                pdf.close()

        if images:
            logger.info(f"Extracted {len(images)} image(s) from {file_path.name}")
        return images

    def _save_image_object(
        self, obj, staging_dir: Path, doc_hash: str, index: int, page_num: int
    ) -> dict | None:
        """Render a single PdfImage object to PNG; return its descriptor or None."""
        try:
            width, height = obj.get_px_size()
            if width < _MIN_IMAGE_DIM or height < _MIN_IMAGE_DIM:
                return None

            pil_image = obj.get_bitmap().to_pil()
            if pil_image.mode not in ("RGB", "RGBA", "L"):
                pil_image = pil_image.convert("RGB")

            image_id = f"{doc_hash}_{index:03d}"
            dest = staging_dir / f"{image_id}.png"
            pil_image.save(str(dest), format="PNG")
            return {"id": image_id, "path": str(dest), "page": page_num}
        except Exception as e:  # pragma: no cover - per-image resilience
            logger.warning(f"Failed to save embedded image #{index}: {e}")
            return None

    @staticmethod
    def _append_image_placeholders(text: str, images: list[dict]) -> str:
        """Append ``[IMAGE: id]`` placeholders so chunker/captioner can attach them."""
        lines = [f"[IMAGE: {img['id']}]" for img in images]
        return f"{text}\n\n## Figures\n" + "\n".join(lines)

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]
