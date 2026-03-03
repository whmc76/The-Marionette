"""CSV output snapshot tests."""
from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import pytest

from src.models import GeneratedComment
from src.csv_writer import to_csv_bytes, write_csv, make_filename, _STANDARD_COLUMNS, _AUDIT_COLUMNS


def _make_comments(n: int = 3) -> list[GeneratedComment]:
    comments = []
    for i in range(n):
        comments.append(GeneratedComment(
            task_id=f"task-{i}",
            text=f"这是第{i+1}条测试评论，字数足够自然流畅",
            category_direction="正向",
            theme="使用体验",
            sub_theme="产品质量",
            persona="真实用户",
            char_count=20,
            validation_status="pass",
            errors=[],
        ))
    return comments


# ── Column structure ─────────────────────────────────────────────

def test_csv_has_all_columns():
    comments = _make_comments(2)
    raw = to_csv_bytes(comments, run_id="test", provider="openai", model="gpt-4o-mini")
    # Strip BOM
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    for col in _STANDARD_COLUMNS + _AUDIT_COLUMNS:
        assert col in fieldnames, f"Missing column: {col}"


def test_csv_row_count():
    n = 5
    comments = _make_comments(n)
    raw = to_csv_bytes(comments)
    text = raw.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    assert len(rows) == n


def test_csv_utf8_bom():
    comments = _make_comments(1)
    raw = to_csv_bytes(comments)
    assert raw[:3] == b"\xef\xbb\xbf", "CSV should start with UTF-8 BOM"


def test_csv_chinese_content():
    comments = _make_comments(1)
    raw = to_csv_bytes(comments)
    text = raw.decode("utf-8-sig")
    assert "这是第1条测试评论" in text


# ── File write ───────────────────────────────────────────────────

def test_write_csv_creates_file():
    comments = _make_comments(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "test_output.csv"
        result_path = write_csv(comments, out_path, run_id="r1", provider="openai", model="m1")
        assert result_path.exists()
        assert result_path.stat().st_size > 0


def test_write_csv_readable_by_csv_reader():
    comments = _make_comments(3)
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.csv"
        write_csv(comments, out_path)
        with open(out_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["评论内容"] == comments[0].text


# ── Filename ─────────────────────────────────────────────────────

def test_make_filename_format():
    name = make_filename("岚图梦想家")
    assert name.startswith("岚图梦想家_")
    assert name.endswith(".csv")
    assert len(name) < 50


def test_make_filename_sanitizes_slashes():
    name = make_filename("产品/型号")
    assert "/" not in name


# ── Validation status in CSV ─────────────────────────────────────

def test_csv_includes_validation_status():
    comments = _make_comments(1)
    comments[0].validation_status = "soft_flag"
    comments[0].errors = ["与其他评论相似"]

    raw = to_csv_bytes(comments)
    text = raw.decode("utf-8-sig")
    assert "soft_flag" in text
    assert "与其他评论相似" in text


# ── Industry audit column ─────────────────────────────────────────

def test_csv_includes_industry_column():
    comments = _make_comments(1)
    raw = to_csv_bytes(comments, industry="beauty")
    text = raw.decode("utf-8-sig")
    assert "industry" in text
    assert "beauty" in text


def test_csv_industry_defaults_to_general():
    comments = _make_comments(1)
    raw = to_csv_bytes(comments)
    text = raw.decode("utf-8-sig")
    assert "general" in text
