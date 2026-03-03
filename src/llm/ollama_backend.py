"""Native Ollama backend via /api/chat — supports think=False for GLM and similar models."""
from __future__ import annotations

import json
import queue as _queue
import threading
import time
from typing import Callable, Optional

import httpx

from src.llm.base import BaseLLMBackend, LLMResponse
from src.llm.openai_backend import (
    _OLLAMA_API_OPTIONS,
    _model_base,
    fetch_ollama_models,
    unload_ollama_model,
)

# Options passed at the request body top level (not inside `options` dict).
# GLM-4.7-Flash: `think` must be at the top level for the native API.
_TOP_LEVEL_OPTIONS = frozenset({"think"})


class OllamaBackend(BaseLLMBackend):
    """Uses Ollama's native /api/chat endpoint.

    Advantages over OpenAI-compatible endpoint:
    - `think=False` is respected properly (disables GLM thinking mode)
    - Avoids OpenAI SDK overhead
    - Streams thinking tokens separately so they don't clog output
    """

    provider = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        ollama_options: Optional[dict] = None,
    ) -> None:
        self.model = model
        self._base = base_url.rstrip("/").removesuffix("/v1")
        raw = ollama_options or {}
        # Separate API-level options from top-level parameters
        self._options: dict = {k: v for k, v in raw.items()
                               if k in _OLLAMA_API_OPTIONS and k not in _TOP_LEVEL_OPTIONS}
        self._top_level: dict = {k: v for k, v in raw.items()
                                 if k in _TOP_LEVEL_OPTIONS}

    # ── Internal helpers ──────────────────────────────────────────

    def _current_ctx(self) -> Optional[int]:
        """Query /api/ps for the currently loaded model's context_length."""
        try:
            r = httpx.get(f"{self._base}/api/ps", timeout=3.0)
            r.raise_for_status()
            target = _model_base(self.model)
            for m in r.json().get("models", []):
                if _model_base(m.get("name", "")) == target:
                    return m.get("context_length")
        except Exception:
            pass
        return None

    def _build_payload(
        self,
        system: str,
        user: str,
        temperature: float,
        extra_options: Optional[dict],
        stream: bool,
    ) -> dict:
        opts = dict(self._options)
        top = dict(self._top_level)

        needed_ctx: int = opts.get("num_ctx", 0)
        if extra_options:
            for k, v in extra_options.items():
                if k == "num_ctx":
                    needed_ctx = max(needed_ctx, v)
                elif k in _TOP_LEVEL_OPTIONS:
                    top[k] = v
                elif k in _OLLAMA_API_OPTIONS:
                    opts[k] = v

        # Avoid model reload: never send num_ctx smaller than currently loaded.
        current_ctx = self._current_ctx()
        if current_ctx is not None:
            opts["num_ctx"] = max(current_ctx, needed_ctx)
        elif needed_ctx > 0:
            opts["num_ctx"] = needed_ctx
        else:
            opts.pop("num_ctx", None)

        # Override temperature via options
        opts["temperature"] = temperature

        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            **top,  # e.g. think=False at top level
        }
        if opts:
            payload["options"] = opts
        return payload

    # ── BaseLLMBackend interface ──────────────────────────────────

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        timeout: float = 30.0,
        extra_options: Optional[dict] = None,
    ) -> LLMResponse:
        payload = self._build_payload(system, user, temperature, extra_options, stream=False)
        r = httpx.post(f"{self._base}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        msg = r.json().get("message", {})
        return LLMResponse(
            content=msg.get("content", ""),
            model=self.model,
            provider="ollama",
        )

    def stream_chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        timeout: float = 30.0,
        on_chunk: Optional[Callable[[str], None]] = None,
        extra_options: Optional[dict] = None,
    ) -> LLMResponse:
        """Streaming via /api/chat with hard wall-clock timeout.

        Thinking tokens (inside <think>…</think>) are consumed silently;
        only actual response content is forwarded to on_chunk.
        """
        payload = self._build_payload(system, user, temperature, extra_options, stream=True)

        result_q: _queue.Queue = _queue.Queue()

        def _worker() -> None:
            try:
                with httpx.stream(
                    "POST",
                    f"{self._base}/api/chat",
                    json=payload,
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        msg = chunk.get("message", {})
                        # thinking=True means this chunk is an internal reasoning token
                        if msg.get("thinking"):
                            continue
                        delta = msg.get("content", "")
                        if delta:
                            result_q.put(("chunk", delta))
                        if chunk.get("done"):
                            break
                result_q.put(("done", None))
            except Exception as exc:
                result_q.put(("error", exc))

        worker = threading.Thread(target=_worker, daemon=True)
        worker.start()

        full = ""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Ollama streaming timed out after {timeout:.0f}s"
                )
            try:
                event_type, value = result_q.get(timeout=min(remaining, 2.0))
            except _queue.Empty:
                continue

            if event_type == "chunk":
                full += value
                if on_chunk:
                    on_chunk(value)
            elif event_type == "done":
                break
            elif event_type == "error":
                raise value  # type: ignore[misc]

        return LLMResponse(content=full, model=self.model, provider="ollama")

    def list_models(self) -> list[str]:
        try:
            return fetch_ollama_models(self._base)
        except Exception:
            return [self.model]

    def test_connection(self) -> bool:
        try:
            r = httpx.get(f"{self._base}/api/tags", timeout=5)
            r.raise_for_status()
            return True
        except Exception:
            return False
