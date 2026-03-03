"""OpenAI-compatible backend (OpenAI, DeepSeek, Moonshot, Ollama)."""
from __future__ import annotations

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

    def _ollama_extra(self, override: Optional[dict] = None) -> Optional[dict]:
        """Return extra_body dict for Ollama requests, merging per-call overrides.

        num_ctx is never downgraded: takes max(stored, override).
        """
        if not self.is_ollama:
            return None
        opts = dict(self._ollama_options)
        if override:
            for k, v in override.items():
                if k == "num_ctx":
                    opts[k] = max(opts.get(k, 0), v)
                else:
                    opts[k] = v
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
        full = ""
        with self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            timeout=timeout,
            stream=True,
            extra_body=self._ollama_extra(extra_options),
        ) as stream:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or "" if chunk.choices else ""
                if delta:
                    full += delta
                    if on_chunk:
                        on_chunk(delta)
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
