"""BaseLLM — abstract interface for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from an LLM chat call."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    All LLM implementations must subclass this and implement the chat method.
    """

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
        """Send messages to the LLM and get a response.

        Args:
            messages: List of ChatMessage objects forming the conversation.
            **kwargs: Additional provider-specific parameters.

        Returns:
            ChatResponse with the model's reply.

        Raises:
            LLMError: If the API call fails.
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'openai', 'azure', 'ollama')."""


class LLMError(Exception):
    """Raised when an LLM API call fails."""

    def __init__(self, message: str, provider: str = "", cause: Exception | None = None):
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}" if provider else message)
