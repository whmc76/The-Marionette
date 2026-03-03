"""Unified LLM client with rate limiting, retry, and failure threshold."""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from src.llm.base import BaseLLMBackend, LLMResponse


class GenerationAborted(Exception):
    """Raised when the global failure threshold is exceeded."""


class LLMClient:
    """Thread-safe client wrapping a backend with retry and failure threshold."""

    def __init__(
        self,
        backend: BaseLLMBackend,
        max_concurrency: int = 5,
        max_retries: int = 2,
        failure_threshold: float = 0.15,
        timeout: float = 30.0,
    ) -> None:
        self._backend = backend
        self._semaphore = threading.Semaphore(max_concurrency)
        self._max_retries = max_retries
        self._failure_threshold = failure_threshold
        self._timeout = timeout

        self._lock = threading.Lock()
        self._total_attempts = 0
        self._total_failures = 0
        self._aborted = False

    # ── Public ──────────────────────────────────────────────────

    def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        extra_options: Optional[dict] = None,
    ) -> LLMResponse:
        """Single synchronous call with retry + failure accounting."""
        if self._aborted:
            raise GenerationAborted("Generation aborted due to high failure rate")

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                with self._semaphore:
                    resp = self._backend.chat(system, user, temperature, self._timeout, extra_options)
                self._record(success=True)
                return resp
            except Exception as exc:
                last_exc = exc
                self._record(success=False)
                if attempt < self._max_retries:
                    # Exponential back-off: 1s, 2s
                    time.sleep(2 ** attempt)

        raise last_exc  # type: ignore[misc]

    def stream_chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.9,
        on_chunk: Optional[Callable[[str], None]] = None,
        extra_options: Optional[dict] = None,
    ) -> LLMResponse:
        """Streaming call with retry + failure accounting."""
        if self._aborted:
            raise GenerationAborted("Generation aborted due to high failure rate")

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                with self._semaphore:
                    resp = self._backend.stream_chat(
                        system, user, temperature, self._timeout, on_chunk, extra_options
                    )
                self._record(success=True)
                return resp
            except Exception as exc:
                last_exc = exc
                self._record(success=False)
                if attempt < self._max_retries:
                    time.sleep(2 ** attempt)

        raise last_exc  # type: ignore[misc]

    def batch_chat(
        self,
        tasks: list[tuple[str, str]],  # list of (system, user)
        temperature: float = 0.9,
        on_progress: Optional[Callable[[int, int, int], None]] = None,
    ) -> list[Optional[LLMResponse]]:
        """
        Parallel batch execution.
        on_progress(success_count, fail_count, retry_count) called after each task.
        Returns list aligned with input tasks (None on failure).
        """
        results: list[Optional[LLMResponse]] = [None] * len(tasks)
        success_count = 0
        fail_count = 0

        with ThreadPoolExecutor(max_workers=self._semaphore._value) as pool:
            future_to_idx = {
                pool.submit(self.chat, s, u, temperature): idx
                for idx, (s, u) in enumerate(tasks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                    success_count += 1
                except GenerationAborted:
                    fail_count += 1
                    break
                except Exception:
                    fail_count += 1
                if on_progress:
                    on_progress(success_count, fail_count, 0)

        return results

    def test_connection(self) -> bool:
        return self._backend.test_connection()

    def list_models(self) -> list[str]:
        return self._backend.list_models()

    @property
    def provider(self) -> str:
        return self._backend.provider

    @property
    def model(self) -> str:
        return getattr(self._backend, "model", "unknown")

    # ── Internal ─────────────────────────────────────────────────

    def _record(self, success: bool) -> None:
        with self._lock:
            self._total_attempts += 1
            if not success:
                self._total_failures += 1
            if (
                self._total_attempts >= 10
                and self._total_failures / self._total_attempts >= self._failure_threshold
            ):
                self._aborted = True
                raise GenerationAborted(
                    f"Failure rate {self._total_failures}/{self._total_attempts} "
                    f"exceeds threshold {self._failure_threshold:.0%}"
                )


def build_client(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    max_concurrency: int = 5,
    max_retries: int = 2,
    failure_threshold: float = 0.15,
    timeout: float = 30.0,
    ollama_options: Optional[dict] = None,
) -> LLMClient:
    """Factory function to build a LLMClient from config."""
    if provider == "anthropic":
        from src.llm.anthropic_backend import AnthropicBackend
        backend: BaseLLMBackend = AnthropicBackend(api_key=api_key, model=model)
    elif provider == "ollama":
        from src.llm.openai_backend import OpenAIBackend
        # Ollama OpenAI-compatible endpoint requires /v1 suffix
        ollama_url = base_url.rstrip("/")
        if not ollama_url.endswith("/v1"):
            ollama_url += "/v1"
        backend = OpenAIBackend(
            api_key="ollama",
            base_url=ollama_url,
            model=model,
            is_ollama=True,
            ollama_options=ollama_options,
        )
        timeout = max(timeout, 120.0)
        max_concurrency = min(max_concurrency, 2)
    else:  # openai / compatible
        from src.llm.openai_backend import OpenAIBackend
        backend = OpenAIBackend(api_key=api_key, base_url=base_url, model=model)

    return LLMClient(
        backend=backend,
        max_concurrency=max_concurrency,
        max_retries=max_retries,
        failure_threshold=failure_threshold,
        timeout=timeout,
    )
