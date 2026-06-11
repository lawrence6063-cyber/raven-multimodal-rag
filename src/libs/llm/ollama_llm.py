"""Ollama LLM implementation — local HTTP endpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError
from src.libs.llm.llm_factory import register_llm

if TYPE_CHECKING:
    from src.core.settings import LLMSettings


@register_llm("ollama")
class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation via HTTP API."""

    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, settings: "LLMSettings"):
        self._settings = settings
        self._model = settings.model
        self._base_url = settings.base_url or self.DEFAULT_BASE_URL
        self._temperature = settings.temperature

    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Send chat request to Ollama HTTP API."""
        import urllib.request
        import urllib.error
        import json

        url = f"{self._base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self._temperature),
            },
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))

            message = body.get("message", {})
            return ChatResponse(
                content=message.get("content", ""),
                model=body.get("model", self._model),
                usage={
                    "prompt_tokens": body.get("prompt_eval_count", 0),
                    "completion_tokens": body.get("eval_count", 0),
                },
                raw=body,
            )
        except urllib.error.URLError as e:
            raise LLMError(
                f"Cannot connect to Ollama at {self._base_url}. Is it running?",
                provider="ollama",
                cause=e,
            ) from e
        except json.JSONDecodeError as e:
            raise LLMError("Invalid JSON response from Ollama.", provider="ollama", cause=e) from e
        except Exception as e:
            raise LLMError(f"Ollama API call failed: {e}", provider="ollama", cause=e) from e

    @property
    def provider_name(self) -> str:
        return "ollama"
