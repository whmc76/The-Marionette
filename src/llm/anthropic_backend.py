"""Anthropic Claude backend."""
from __future__ import annotations

from typing import Callable, Optional

import anthropic

from src.llm.base import BaseLLMBackend, LLMResponse


_KNOWN_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


class AnthropicBackend(BaseLLMBackend):
    provider = "anthropic"

    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001") -> None:
        self.model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        timeout: float = 30.0,
        extra_options: Optional[dict] = None,  # noqa: ARG002 — Anthropic handles ctx server-side
    ) -> LLMResponse:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=timeout,
        )
        content = "".join(
            block.text for block in resp.content if hasattr(block, "text")
        )
        return LLMResponse(
            content=content,
            model=self.model,
            provider="anthropic",
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )

    def stream_chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        timeout: float = 30.0,
        on_chunk: Optional[Callable[[str], None]] = None,
        extra_options: Optional[dict] = None,  # noqa: ARG002 — Anthropic handles ctx server-side
    ) -> LLMResponse:
        """Stream chat using raw events to capture thinking blocks.

        Thinking block deltas are emitted as <think>…</think> so the caller
        can distinguish them from regular output tokens.
        """
        full_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        current_block_type: Optional[str] = None

        with self._client.messages.stream(
            model=self.model,
            max_tokens=2048,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            timeout=timeout,
        ) as stream:
            for event in stream:
                etype = event.type
                if etype == "content_block_start":
                    current_block_type = event.content_block.type
                    if current_block_type == "thinking" and on_chunk:
                        on_chunk("<think>")
                elif etype == "content_block_delta":
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        if on_chunk:
                            on_chunk(delta.thinking)
                    elif delta.type == "text_delta":
                        full_text += delta.text
                        if on_chunk:
                            on_chunk(delta.text)
                elif etype == "content_block_stop":
                    if current_block_type == "thinking" and on_chunk:
                        on_chunk("</think>")
                    current_block_type = None
            try:
                final = stream.get_final_message()
                if final and final.usage:
                    prompt_tokens = final.usage.input_tokens
                    completion_tokens = final.usage.output_tokens
            except Exception:
                pass

        return LLMResponse(
            content=full_text,
            model=self.model,
            provider="anthropic",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def list_models(self) -> list[str]:
        return _KNOWN_MODELS

    def test_connection(self) -> bool:
        try:
            self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
                timeout=15,
            )
            return True
        except Exception:
            return False
