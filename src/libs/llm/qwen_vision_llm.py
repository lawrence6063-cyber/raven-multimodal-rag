"""Qwen Vision LLM implementation — qwen-vl-max via DashScope compatible-mode.

通义千问视觉模型（qwen-vl-max / qwen-vl-plus）通过阿里云 DashScope 提供，兼容
OpenAI SDK 的 ``image_url`` 多模态消息格式，仅 ``base_url`` 不同：
https://dashscope.aliyuncs.com/compatible-mode/v1

因此复用标准 ``openai.OpenAI`` 客户端即可，与 ``QwenLLM`` / ``QwenEmbedding``
共用同一个 DashScope API Key。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import TYPE_CHECKING

from src.libs.llm.base_vision_llm import BaseVisionLLM
from src.libs.llm.base_llm import ChatResponse, LLMError
from src.libs.llm.vision_llm_factory import register_vision_llm

if TYPE_CHECKING:
    from src.core.settings import VisionLLMSettings


@register_vision_llm("qwen_vision")
class QwenVisionLLM(BaseVisionLLM):
    """Qwen-VL vision model via Alibaba DashScope (OpenAI-compatible)."""

    # DEFAULT_BASE_URL DashScope OpenAI-compatible endpoint
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # DEFAULT_MODEL default Qwen vision model
    DEFAULT_MODEL = "qwen-vl-max"

    def __init__(self, settings: "VisionLLMSettings"):
        self._settings = settings
        self._model = settings.model or self.DEFAULT_MODEL
        self._api_key = settings.api_key
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL
        self._max_image_size = settings.max_image_size

    def chat_with_image(self, text: str, image_path: str | bytes) -> ChatResponse:
        """Send text + image to the Qwen vision model.

        Args:
            text: Text prompt describing what to analyze.
            image_path: Path to an image file, or raw image bytes.

        Returns:
            ChatResponse with the model's description.

        Raises:
            LLMError: If the API call fails.
        """
        import openai

        image_b64 = self._prepare_image(image_path)

        try:
            client = openai.OpenAI(api_key=self._api_key, base_url=self._base_url)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
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
            raise LLMError("Qwen Vision auth failed", provider="qwen_vision", cause=e) from e
        except Exception as e:
            raise LLMError(f"Qwen Vision API failed: {e}", provider="qwen_vision", cause=e) from e

    def _prepare_image(self, image_path: str | bytes) -> str:
        """Convert image to base64, downscaling when it exceeds the size limit."""
        if isinstance(image_path, bytes):
            image_data = image_path
        else:
            path = Path(image_path)
            if not path.exists():
                raise LLMError(f"Image not found: {path}", provider="qwen_vision")
            image_data = path.read_bytes()

        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_data))
            max_dim = self._max_image_size
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), Image.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                image_data = buf.getvalue()
        except ImportError:
            pass  # PIL not available, send original bytes

        return base64.b64encode(image_data).decode("utf-8")

    @property
    def provider_name(self) -> str:
        return "qwen_vision"
