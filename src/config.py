"""Configuration management: .env for secrets, config.json for non-sensitive settings."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent
_CONFIG_FILE = _ROOT / "config.json"
_ENV_FILE = _ROOT / ".env"

# Load .env on import
load_dotenv(_ENV_FILE, override=False)


def _load_json() -> dict[str, Any]:
    if _CONFIG_FILE.exists():
        try:
            return json.loads(_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_json(data: dict[str, Any]) -> None:
    _CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── LLM Settings dataclass (persisted to config.json) ────────────

@dataclass
class LLMSettings:
    """Non-sensitive LLM configuration. Persisted to config.json."""
    provider: str = "openai"
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.9
    max_concurrency: int = 5
    batch_size: int = 8
    max_retries: int = 2
    failure_threshold: float = 0.15
    # Ollama VRAM optimisation (passed via options in each request)
    ollama_num_ctx: int = 8192          # context window; default can be 128K → OOM
    ollama_num_gpu: int = 99            # GPU layers; 99 = offload all to GPU
    ollama_flash_attn: bool = True      # flash attention — cuts peak VRAM
    ollama_kv_cache_type: str = "q8_0" # KV-cache dtype: f16 / q8_0 / q4_0
    # Runtime-only (not persisted)
    conn_verified: bool = field(default=False, repr=False)

    _CONFIG_KEY = "llm"

    def provider_label(self) -> str:
        return {"openai": "OpenAI / 兼容", "anthropic": "Claude", "ollama": "Ollama"}.get(
            self.provider, self.provider
        )


# ---------- Public API ----------

class Config:
    """Thin wrapper that merges .env secrets + config.json non-sensitive settings."""

    _defaults: dict[str, Any] = {
        "output_dir": str(_ROOT / "output"),
    }

    def __init__(self) -> None:
        self._file = _load_json()

    def get(self, key: str, fallback: Any = None) -> Any:
        # Priority: env > config.json > defaults
        env_val = os.environ.get(key.upper())
        if env_val is not None:
            return env_val
        return self._file.get(key, self._defaults.get(key, fallback))

    def set(self, key: str, value: Any) -> None:
        """Persist non-sensitive setting to config.json."""
        self._file[key] = value
        _save_json(self._file)

    # ── LLM settings ─────────────────────────────────────────────

    def load_llm_settings(self) -> LLMSettings:
        """Load LLM settings from config.json, falling back to defaults."""
        raw: dict = self._file.get("llm", {})
        d = LLMSettings()
        return LLMSettings(
            provider=raw.get("provider", d.provider),
            base_url=raw.get("base_url", d.base_url),
            model=raw.get("model", d.model),
            temperature=raw.get("temperature", d.temperature),
            max_concurrency=raw.get("max_concurrency", d.max_concurrency),
            batch_size=raw.get("batch_size", d.batch_size),
            max_retries=raw.get("max_retries", d.max_retries),
            failure_threshold=raw.get("failure_threshold", d.failure_threshold),
            ollama_num_ctx=raw.get("ollama_num_ctx", d.ollama_num_ctx),
            ollama_num_gpu=raw.get("ollama_num_gpu", d.ollama_num_gpu),
            ollama_flash_attn=raw.get("ollama_flash_attn", d.ollama_flash_attn),
            ollama_kv_cache_type=raw.get("ollama_kv_cache_type", d.ollama_kv_cache_type),
            conn_verified=raw.get("conn_verified", False),
        )

    def save_llm_settings(self, s: LLMSettings) -> None:
        """Persist LLM settings to config.json (API keys excluded)."""
        self._file["llm"] = {
            "provider": s.provider,
            "base_url": s.base_url,
            "model": s.model,
            "temperature": s.temperature,
            "max_concurrency": s.max_concurrency,
            "batch_size": s.batch_size,
            "max_retries": s.max_retries,
            "failure_threshold": s.failure_threshold,
            "ollama_num_ctx": s.ollama_num_ctx,
            "ollama_num_gpu": s.ollama_num_gpu,
            "ollama_flash_attn": s.ollama_flash_attn,
            "ollama_kv_cache_type": s.ollama_kv_cache_type,
            "conn_verified": s.conn_verified,
        }
        _save_json(self._file)

    # ── Secrets (never written to disk) ──────────────────────────

    @property
    def openai_api_key(self) -> Optional[str]:
        return os.environ.get("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.environ.get("ANTHROPIC_API_KEY")

    def env_api_key(self, provider: str) -> str:
        return {
            "openai": self.openai_api_key or "",
            "anthropic": self.anthropic_api_key or "",
            "ollama": "",
        }.get(provider, "")

    def set_session_key(self, provider: str, key: str) -> None:
        """Inject API key into process env for this session."""
        env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
        if env_name:
            os.environ[env_name] = key

    def ensure_output_dir(self) -> Path:
        d = Path(self.get("output_dir"))
        d.mkdir(parents=True, exist_ok=True)
        return d


# Module-level singleton
config = Config()
