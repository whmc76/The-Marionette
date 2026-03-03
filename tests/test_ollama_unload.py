"""Tests for Ollama model unload correctness.

Three bugs investigated:
  1. unload triggers a NEW model load when model isn't running (VRAM spike).
  2. Model name tag mismatch silently skips the unload.
  3. OLLAMA_MAX_LOADED_MODELS > 1 keeps multiple models in VRAM at once.
"""
import pytest
from unittest.mock import patch, MagicMock
from src.llm.openai_backend import unload_ollama_model


# ── helpers ──────────────────────────────────────────────────────────────────

def _ps_response(model_names: list[str]):
    """Fake /api/ps response listing the given model names as running."""
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    mock.json.return_value = {"models": [{"name": n} for n in model_names]}
    return mock


def _ok_response():
    mock = MagicMock()
    mock.raise_for_status = MagicMock()
    return mock


# ── Bug 1: unload must NOT trigger a model load ───────────────────────────────

def test_unload_skips_post_when_model_not_running():
    """If the model is NOT in /api/ps, we must NOT POST to /api/generate.

    Posting keep_alive=0 to /api/generate for an unloaded model causes Ollama
    to LOAD the model first (occupying VRAM), then immediately unload it.
    This is the primary reason VRAM stays full after switching models.
    """
    with patch("httpx.get", return_value=_ps_response([])) as mock_get, \
         patch("httpx.post") as mock_post:
        result = unload_ollama_model("http://localhost:11434", "glm4.7:latest")

    mock_get.assert_called_once()               # /api/ps was checked
    mock_post.assert_not_called()               # /api/generate was NOT called
    assert result is True                       # "already unloaded" = success


def test_unload_sends_post_when_model_is_running():
    """When model IS in /api/ps, we must POST keep_alive=0 to unload it."""
    with patch("httpx.get", return_value=_ps_response(["glm4.7:latest"])), \
         patch("httpx.post", return_value=_ok_response()) as mock_post:
        result = unload_ollama_model("http://localhost:11434", "glm4.7:latest")

    mock_post.assert_called_once()
    body = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json") or mock_post.call_args[0][1]
    assert body == {"model": "glm4.7:latest", "keep_alive": 0}
    assert result is True


# ── Bug 2: model name tag mismatch ───────────────────────────────────────────

def test_unload_matches_without_tag_suffix():
    """Ollama /api/ps may list 'glm4.7:latest' while LLMSettings stores 'glm4.7'.
    The lookup must match the base name (before ':') so unload is not skipped.
    """
    # /api/ps returns the full tagged name
    with patch("httpx.get", return_value=_ps_response(["glm4.7:latest"])), \
         patch("httpx.post", return_value=_ok_response()) as mock_post:
        # LLMSettings stored without tag
        result = unload_ollama_model("http://localhost:11434", "glm4.7")

    mock_post.assert_called_once()   # unload was triggered despite missing tag
    assert result is True


def test_unload_matches_with_extra_tag():
    """Reverse case: settings stored 'glm4.7:latest', /api/ps lists 'glm4.7'."""
    with patch("httpx.get", return_value=_ps_response(["glm4.7"])), \
         patch("httpx.post", return_value=_ok_response()) as mock_post:
        result = unload_ollama_model("http://localhost:11434", "glm4.7:latest")

    mock_post.assert_called_once()
    assert result is True


# ── URL normalisation sanity checks ──────────────────────────────────────────

def test_unload_strips_v1_suffix_from_url():
    """/api/ps and /api/generate must use the root URL, not the /v1 path."""
    with patch("httpx.get", return_value=_ps_response(["m:latest"])) as mock_get, \
         patch("httpx.post", return_value=_ok_response()) as mock_post:
        unload_ollama_model("http://localhost:11434/v1", "m:latest")

    get_url = mock_get.call_args[0][0]
    post_url = mock_post.call_args[0][0]
    assert "/v1" not in get_url,  f"GET URL must not contain /v1: {get_url}"
    assert "/v1" not in post_url, f"POST URL must not contain /v1: {post_url}"


def test_unload_returns_false_on_ps_failure():
    """If /api/ps fails, return False without attempting unload."""
    with patch("httpx.get", side_effect=Exception("timeout")), \
         patch("httpx.post") as mock_post:
        result = unload_ollama_model("http://localhost:11434", "glm4.7:latest")

    mock_post.assert_not_called()
    assert result is False
