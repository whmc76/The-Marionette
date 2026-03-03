"""Generator pipeline integration tests with mock LLM."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from src.models import BriefSpec, CommentCategory, CommentTask
from src.llm.base import LLMResponse
from src.llm.client import LLMClient
from src.generator import schedule_tasks, run_generation, _parse_numbered_list


def _make_spec(num_categories: int = 2) -> BriefSpec:
    cats = []
    for i in range(num_categories):
        direction = "正向" if i % 2 == 0 else "反击"
        cats.append(CommentCategory(
            direction=direction,
            theme=f"主题{i+1}",
            sub_themes=["子主题A", "子主题B"],
            description="测试描述",
            personas=["普通车主", "科技爱好者"],
            example_comments=["这辆车真的太棒了", "续航表现非常出色"],
        ))

    return BriefSpec(
        title="测试Brief",
        product_name="测试汽车",
        product_background="一款优秀的电动汽车",
        general_rules=["自然口语化", "贴近真实用户"],
        forbidden_phrases=["最优惠", "第一名"],
        categories=cats,
        positive_ratio=0.5,
        negative_ratio=0.5,
        min_char_length=15,
        platform_targets=["微博", "小红书"],
    )


def _mock_client(response_text: str) -> LLMClient:
    backend = MagicMock()
    backend.provider = "mock"
    backend.model = "mock-model"

    # run_generation now uses stream_chat; simulate by calling on_chunk once.
    def _fake_stream_chat(system, user, temperature=0.9, timeout=30.0,
                          on_chunk=None, extra_options=None):
        if on_chunk:
            on_chunk(response_text)
        return LLMResponse(content=response_text, model="mock-model", provider="mock")

    backend.stream_chat.side_effect = _fake_stream_chat
    client = LLMClient(backend=backend, max_concurrency=1)
    return client


# ── schedule_tasks tests ──────────────────────────────────────────

def test_schedule_tasks_total_count():
    spec = _make_spec(2)
    tasks = schedule_tasks(spec, total_count=20)
    total = sum(t.target_count for t in tasks)
    assert total == 20


def test_schedule_tasks_respects_ratio():
    spec = _make_spec(2)
    spec.positive_ratio = 0.7
    spec.negative_ratio = 0.3
    tasks = schedule_tasks(spec, total_count=100)

    pos_total = sum(t.target_count for t in tasks if t.category.direction == "正向")
    neg_total = sum(t.target_count for t in tasks if t.category.direction != "正向")
    assert pos_total == pytest.approx(70, abs=2)
    assert neg_total == pytest.approx(30, abs=2)


def test_schedule_tasks_batch_size():
    spec = _make_spec(1)
    tasks = schedule_tasks(spec, total_count=30, batch_size=8)
    for t in tasks:
        assert t.target_count <= 8


def test_schedule_tasks_empty_categories():
    spec = _make_spec(0)
    tasks = schedule_tasks(spec, total_count=10)
    assert tasks == []


# ── _parse_numbered_list tests ────────────────────────────────────

def test_parse_numbered_list_standard():
    text = "1. 这是第一条评论\n2. 这是第二条评论\n3. 第三条"
    result = _parse_numbered_list(text)
    assert result == ["这是第一条评论", "这是第二条评论", "第三条"]


def test_parse_numbered_list_mixed():
    text = "1. 正常评论一\n2、带顿号的评论\n3。带句号的"
    result = _parse_numbered_list(text)
    assert len(result) == 3


def test_parse_numbered_list_empty():
    assert _parse_numbered_list("") == []


# ── run_generation tests ─────────────────────────────────────────

def test_run_generation_basic():
    spec = _make_spec(2)
    tasks = schedule_tasks(spec, total_count=10, batch_size=5)

    # Mock returns 5 numbered comments
    mock_response = "\n".join(f"{i+1}. 这是一条很好的测试评论，字数足够多" for i in range(5))
    client = _mock_client(mock_response)

    result = run_generation(spec, tasks, client, temperature=0.9)
    assert len(result.comments) > 0
    assert result.progress.success > 0


def test_run_generation_records_metadata():
    spec = _make_spec(1)
    tasks = schedule_tasks(spec, total_count=5, batch_size=5)
    mock_response = "\n".join(f"{i+1}. 评论内容测试文字足够长的一条评论" for i in range(5))
    client = _mock_client(mock_response)

    result = run_generation(spec, tasks, client)
    for comment in result.comments:
        assert comment.task_id
        assert comment.category_direction in ("正向", "反击", "引导")
        assert comment.char_count > 0


def test_run_generation_progress_callback():
    spec = _make_spec(2)
    tasks = schedule_tasks(spec, total_count=10, batch_size=5)
    mock_response = "\n".join(f"{i+1}. 这是测试评论内容足够长度的评论文字" for i in range(5))
    client = _mock_client(mock_response)

    progress_calls = []
    run_generation(spec, tasks, client, on_progress=lambda p: progress_calls.append(p.done))

    assert len(progress_calls) > 0
    assert progress_calls[-1] > 0
