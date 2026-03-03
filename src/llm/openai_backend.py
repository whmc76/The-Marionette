"""OpenAI-compatible backend (OpenAI, DeepSeek, Moonshot, Ollama)."""
from __future__ import annotations

import queue as _queue
import threading
import time
from typing import Callable, Optional

import httpx
import openai

from src.llm.base import BaseLLMBackend, LLMResponse


def fetch_ollama_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """Fetch available model names from Ollama /api/tags. Raises on failure."""
    root = base_url.rstrip("/").removesuffix("/v1")
    r = httpx.get(f"{root}/api/tags", timeout=timeout)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def _model_base(name: str) -> str:
    """Return model name without the ':tag' suffix for loose matching."""
    return name.split(":")[0]


def unload_ollama_model(base_url: str, model: str, timeout: float = 10.0) -> bool:
    """Ask Ollama to evict a model from VRAM immediately (keep_alive=0).

    Steps:
      1. GET /api/ps — list currently loaded models.
      2. If the model is NOT running, return True immediately.
         Skipping this check causes Ollama to LOAD the model first
         (occupying VRAM) before it can apply keep_alive=0.
      3. POST /api/generate with keep_alive=0 to actually unload.

    Name matching is loose: 'glm4.7' matches 'glm4.7:latest' and vice versa.
    Returns True if model was unloaded or wasn't loaded; False on error.
    """
    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        ps = httpx.get(f"{root}/api/ps", timeout=5.0)
        ps.raise_for_status()
        running = [m["name"] for m in ps.json().get("models", [])]
        target_base = _model_base(model)
        if not any(_model_base(n) == target_base for n in running):
            return True  # already unloaded, nothing to do

        r = httpx.post(
            f"{root}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=timeout,
        )
        r.raise_for_status()
        return True
    except Exception:
        return False


# Parameters accepted by Ollama's /v1/chat/completions `options` field.
# flash_attn and kv_cache_type are Ollama *env vars* (OLLAMA_FLASH_ATTENTION,
# OLLAMA_KV_CACHE_TYPE), NOT per-request options — sending them hangs the call.
_OLLAMA_API_OPTIONS = frozenset({
    "num_ctx", "num_batch", "num_gpu", "num_thread",
    "seed", "temperature", "top_k", "top_p", "min_p",
    "repeat_last_n", "repeat_penalty",
    "mirostat", "mirostat_eta", "mirostat_tau",
    "tfs_z", "num_predict", "stop",
    "think",  # GLM-4.7-Flash: False disables thinking mode for faster structured extraction
})


class OpenAIBackend(BaseLLMBackend):
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        is_ollama: bool = False,
        ollama_options: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.is_ollama = is_ollama
        # Store only the options Ollama actually accepts as API parameters.
        raw = ollama_options or {}
        self._ollama_options: dict = {k: v for k, v in raw.items() if k in _OLLAMA_API_OPTIONS}
        self._client = openai.OpenAI(
            api_key=api_key or "ollama",  # Ollama doesn't need a real key
            base_url=base_url,
        )

    def _current_ollama_ctx(self) -> Optional[int]:
        """Query Ollama /api/ps for the currently loaded model's context_length.

        Returns None if the model is not loaded or the query fails.
        This is used to avoid forcing a model reload when the current
        context is already sufficient for the request.
        """
        try:
            base = str(self._client.base_url).rstrip("/").removesuffix("/v1")
            r = httpx.get(f"{base}/api/ps", timeout=3.0)
            r.raise_for_status()
            target = _model_base(self.model)
            for m in r.json().get("models", []):
                if _model_base(m.get("name", "")) == target:
                    return m.get("context_length")
        except Exception:
            pass
        return None

    def _ollama_extra(self, override: Optional[dict] = None) -> Optional[dict]:
        """Return extra_body dict for Ollama requests, merging per-call overrides.

        Avoids triggering model reloads: if the model is already loaded, we send
        num_ctx = max(current_loaded_ctx, needed_ctx) so Ollama never needs to
        downgrade (which requires a full reload that can freeze for several minutes).
        Only upgrades num_ctx when the document genuinely needs more context.
        """
        if not self.is_ollama:
            return None
        opts = dict(self._ollama_options)
        needed_ctx: int = opts.get("num_ctx", 0)
        if override:
            for k, v in override.items():
                if k == "num_ctx":
                    needed_ctx = max(needed_ctx, v)
                else:
                    opts[k] = v

        # Check what context the model is currently loaded with.
        # Sending num_ctx < current_loaded causes a full model reload (minutes).
        # Sending num_ctx = current_loaded keeps the model as-is (instant).
        # Sending num_ctx > current_loaded upgrades context (reload, but necessary).
        current_ctx = self._current_ollama_ctx()
        if current_ctx is not None:
            opts["num_ctx"] = max(current_ctx, needed_ctx)
        elif needed_ctx > 0:
            opts["num_ctx"] = needed_ctx
        else:
            opts.pop("num_ctx", None)

        return {"options": opts} if opts else None

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        timeout: float = 30.0,
        extra_options: Optional[dict] = None,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            timeout=timeout,
            extra_body=self._ollama_extra(extra_options),
        )
        choice = resp.choices[0]
        usage = resp.usage
        return LLMResponse(
            content=choice.message.content or "",
            model=self.model,
            provider="ollama" if self.is_ollama else "openai",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
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
        """Streaming chat with a hard wall-clock timeout.

        Uses a worker thread so that the timeout applies to the total
        request duration, not just the per-chunk read interval.  This
        prevents Ollama keep-alive SSE frames from resetting the timer
        indefinitely during long model thinking phases.
        """
        result_q: _queue.Queue = _queue.Queue()
        extra_body = self._ollama_extra(extra_options)

        def _worker() -> None:
            try:
                with self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature,
                    timeout=timeout,
                    stream=True,
                    extra_body=extra_body,
                ) as stream:
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
                        if delta:
                            result_q.put(("chunk", delta))
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
                    f"LLM streaming request timed out after {timeout:.0f}s "
                    f"(no response from model)"
                )
            try:
                event_type, value = result_q.get(timeout=min(remaining, 2.0))
            except _queue.Empty:
                continue  # re-check deadline on next iteration

            if event_type == "chunk":
                full += value
                if on_chunk:
                    on_chunk(value)
            elif event_type == "done":
                break
            elif event_type == "error":
                raise value  # type: ignore[misc]

        return LLMResponse(
            content=full,
            model=self.model,
            provider="ollama" if self.is_ollama else "openai",
        )

    def list_models(self) -> list[str]:
        if self.is_ollama:
            return self._list_ollama_models()
        try:
            models = self._client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return [self.model]

    def _list_ollama_models(self) -> list[str]:
        """Use Ollama /api/tags endpoint."""
        try:
            base = str(self._client.base_url).rstrip("/").removesuffix("/v1")
            r = httpx.get(f"{base}/api/tags", timeout=5)
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return [self.model]

    def test_connection(self) -> bool:
        if self.is_ollama:
            # For Ollama, just verify the service is reachable via /api/tags.
            # Avoid inference — loading a model into VRAM can take 30+ seconds.
            try:
                base = str(self._client.base_url).rstrip("/").removesuffix("/v1")
                r = httpx.get(f"{base}/api/tags", timeout=5)
                r.raise_for_status()
                return True
            except Exception:
                return False
        # For OpenAI-compatible APIs, do a minimal 1-token completion.
        try:
            self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                timeout=15,
            )
            return True
        except Exception:
            return False
