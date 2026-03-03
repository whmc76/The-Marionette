"""Abstract base class for LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BaseLLMBackend(ABC):
    provider: str = "base"

    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        timeout: float = 30.0,
        extra_options: Optional[dict] = None,
    ) -> LLMResponse:
        """Synchronous chat completion. Raises on failure."""
        ...

    def stream_chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        timeout: float = 30.0,
        on_chunk: Optional[Callable[[str], None]] = None,
        extra_options: Optional[dict] = None,
    ) -> LLMResponse:
        """Streaming chat. Default falls back to chat() if not overridden."""
        resp = self.chat(system, user, temperature, timeout, extra_options)
        if on_chunk:
            on_chunk(resp.content)
        return resp

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return available model IDs."""
        ...

    @abstractmethod
    def test_connection(self) -> bool:
        """Return True if backend is reachable."""
        ...
