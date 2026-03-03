"""Tests for Ollama extra_body options filtering."""
import pytest
from unittest.mock import MagicMock, patch, call
from src.llm.openai_backend import OpenAIBackend
from src.llm.base import LLMResponse


def _make_backend(ollama_options: dict) -> OpenAIBackend:
    backend = OpenAIBackend(
        api_key="ollama",
        base_url="http://localhost:11434/v1",
        model="glm4.7:latest",
        is_ollama=True,
        ollama_options=ollama_options,
    )
    return backend


def _fake_completion(content: str = "{}"):
    """Build a minimal mock completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _fake_stream_chunk(content: str):
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = content
    return chunk


# ── Bug reproduction: flash_attn / kv_cache_type are NOT valid API options ─────

def test_invalid_options_not_sent_via_api():
    """flash_attn and kv_cache_type must NOT appear in extra_body options.

    Sending them to Ollama causes the request to hang indefinitely because
    Ollama does not recognise them as valid model parameters.
    They are environment variables (OLLAMA_FLASH_ATTENTION, OLLAMA_KV_CACHE_TYPE)
    and must be set before starting the Ollama service.
    """
    backend = _make_backend({
        "num_ctx": 8192,
        "num_gpu": 99,
        "flash_attn": True,       # env var — must be stripped
        "kv_cache_type": "q8_0",  # env var — must be stripped
    })

    with patch.object(backend._client.chat.completions, "create",
                      return_value=_fake_completion()) as mock_create:
        backend.chat(system="sys", user="user", temperature=0.1, timeout=30)

    _, kwargs = mock_create.call_args
    extra = kwargs.get("extra_body") or {}
    options = extra.get("options", {})

    assert "flash_attn" not in options, (
        "flash_attn must NOT be sent via API — it's an env var "
        "(OLLAMA_FLASH_ATTENTION) and causes Ollama to hang when unknown options are received"
    )
    assert "kv_cache_type" not in options, (
        "kv_cache_type must NOT be sent via API — it's an env var "
        "(OLLAMA_KV_CACHE_TYPE) and causes Ollama to hang when unknown options are received"
    )


def test_valid_options_are_sent():
    """num_ctx and num_gpu must be forwarded to Ollama via extra_body."""
    backend = _make_backend({
        "num_ctx": 8192,
        "num_gpu": 99,
        "flash_attn": True,
        "kv_cache_type": "q8_0",
    })

    with patch.object(backend._client.chat.completions, "create",
                      return_value=_fake_completion()) as mock_create:
        backend.chat(system="sys", user="user", temperature=0.1, timeout=30)

    _, kwargs = mock_create.call_args
    extra = kwargs.get("extra_body") or {}
    options = extra.get("options", {})

    assert options.get("num_ctx") == 8192, "num_ctx must be forwarded to Ollama"
    assert options.get("num_gpu") == 99,   "num_gpu must be forwarded to Ollama"


def test_stream_chat_valid_options_only():
    """stream_chat must also filter out invalid options."""
    backend = _make_backend({
        "num_ctx": 4096,
        "flash_attn": True,   # must be stripped
    })

    chunk = _fake_stream_chunk("hello")
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=iter([chunk]))
    mock_stream.__exit__ = MagicMock(return_value=False)

    with patch.object(backend._client.chat.completions, "create",
                      return_value=mock_stream) as mock_create:
        backend.stream_chat(system="sys", user="user", temperature=0.1)

    _, kwargs = mock_create.call_args
    extra = kwargs.get("extra_body") or {}
    options = extra.get("options", {})

    assert "flash_attn" not in options
    assert options.get("num_ctx") == 4096


def test_non_ollama_sends_no_extra_body():
    """Non-Ollama backends must never send extra_body."""
    backend = OpenAIBackend(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        is_ollama=False,
    )
    with patch.object(backend._client.chat.completions, "create",
                      return_value=_fake_completion()) as mock_create:
        backend.chat(system="sys", user="user")

    _, kwargs = mock_create.call_args
    extra_body = kwargs.get("extra_body")
    assert not extra_body, "Non-Ollama backend must not send extra_body"
