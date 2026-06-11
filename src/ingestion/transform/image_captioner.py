"""ImageCaptioner — generates text descriptions for images using Vision LLM."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.core.types import Chunk
from src.ingestion.transform.base_transform import BaseTransform

if TYPE_CHECKING:
    from src.core.settings import Settings


class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks."""

    def __init__(self, settings: "Settings"):
        self._enabled = settings.vision_llm.enabled
        self._settings = settings

    def transform(self, chunks: list[Chunk]) -> list[Chunk]:
        """Add image captions to chunks containing image references."""
        if not self._enabled:
            return chunks

        result = []
        for chunk in chunks:
            image_refs = chunk.metadata.get("image_refs", [])
            images = chunk.metadata.get("images", [])

            if not image_refs or not images:
                result.append(chunk)
                continue

            try:
                captions = self._generate_captions(images)
                # Append captions to chunk text
                caption_text = "\n".join(f"[Image {img['id']}]: {cap}" for img, cap in zip(images, captions) if cap)
                new_text = chunk.text + "\n\n" + caption_text if caption_text else chunk.text
                new_meta = dict(chunk.metadata)
                new_meta["image_captions"] = dict(zip([img["id"] for img in images], captions))
                result.append(Chunk(id=chunk.id, text=new_text, metadata=new_meta, source_ref=chunk.source_ref))
            except Exception:
                chunk.metadata["has_unprocessed_images"] = True
                result.append(chunk)

        return result

    def _generate_captions(self, images: list[dict]) -> list[str]:
        """Generate captions for images using the configured Vision LLM."""
        from src.libs.llm.vision_llm_factory import VisionLLMFactory

        vision_llm = VisionLLMFactory.create(self._settings.vision_llm)
        prompt_path = Path("config/prompts/image_captioning.txt")
        prompt_text = prompt_path.read_text(encoding="utf-8") if prompt_path.exists() else "Describe this image."

        captions = []
        for img in images:
            img_path = img.get("path", "")
            if img_path and Path(img_path).exists():
                try:
                    response = vision_llm.chat_with_image(prompt_text, img_path)
                    captions.append(response.content)
                except Exception:
                    captions.append("")
            else:
                captions.append("")

        return captions

    @property
    def name(self) -> str:
        return "image_captioner"
