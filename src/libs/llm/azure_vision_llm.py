"""Azure Vision LLM implementation — GPT-4o / GPT-4-Vision via Azure OpenAI."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.base_llm import ChatResponse, LLMError

if TYPE_CHECKING:
    from src.core.settings import VisionLLMSettings


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI Vision LLM (GPT-4o / GPT-4-Vision-Preview)."""

    def __init__(self, settings: "VisionLLMSettings"):
        self._settings = settings
        self._model = settings.model
        self._api_key = settings.api_key
        self._azure_endpoint = settings.azure_endpoint
        self._api_version = settings.api_version
        self._max_image_size = settings.max_image_size

    def chat_with_image(self, text: str, image_path: str | bytes) -> ChatResponse:
        """Send text + image to Azure Vision LLM."""
        import openai

        image_b64 = self._prepare_image(image_path)

        try:
            client = openai.AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._azure_endpoint,
                api_version=self._api_version,
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ]

            response = client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=1024,
            )

            choice = response.choices[0]
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return ChatResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                raw=response,
            )

        except openai.AuthenticationError as e:
            raise LLMError("Azure Vision auth failed.", provider="azure_vision", cause=e) from e
        except Exception as e:
            raise LLMError(f"Azure Vision API failed: {e}", provider="azure_vision", cause=e) from e

    def _prepare_image(self, image_path: str | bytes) -> str:
        """Convert image to base64, resizing if needed."""
        if isinstance(image_path, bytes):
            image_data = image_path
        else:
            path = Path(image_path)
            if not path.exists():
                raise LLMError(f"Image not found: {path}", provider="azure_vision")
            image_data = path.read_bytes()

        # Optional: resize if too large (requires PIL)
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(image_data))
            max_dim = self._max_image_size
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                image_data = buf.getvalue()
        except ImportError:
            pass  # PIL not available, skip resize

        return base64.b64encode(image_data).decode("utf-8")

    @property
    def provider_name(self) -> str:
        return "azure_vision"
