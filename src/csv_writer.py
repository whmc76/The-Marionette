"""CSV output with audit columns. UTF-8 BOM for Excel compatibility."""
from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime
from pathlib import Path

from src.models import BriefSpec, GeneratedComment


_STANDARD_COLUMNS = [
    "序号",
    "评论内容",
    "分类方向",
    "主题",
    "子主题",
    "人设",
    "情感倾向",
    "字数",
    "生成时间",
]

_AUDIT_COLUMNS = [
    "run_id",
    "provider",
    "model",
    "prompt_version",
    "validation_status",
    "errors",
]


def _sentiment(direction: str) -> str:
    mapping = {"正向": "正面", "反击": "负面", "引导": "中性"}
    return mapping.get(direction, "中性")


def _build_rows(
    comments: list[GeneratedComment],
    run_id: str,
    provider: str,
    model: str,
    prompt_version: str,
    timestamp: str,
) -> list[dict]:
    rows = []
    for i, c in enumerate(comments, start=1):
        row = {
            "序号": i,
            "评论内容": c.text,
            "分类方向": c.category_direction,
            "主题": c.theme,
            "子主题": c.sub_theme,
            "人设": c.persona,
            "情感倾向": _sentiment(c.category_direction),
            "字数": c.char_count,
            "生成时间": timestamp,
            "run_id": run_id,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "validation_status": c.validation_status,
            "errors": "; ".join(c.errors),
        }
        rows.append(row)
    return rows


def write_csv(
    comments: list[GeneratedComment],
    output_path: Path,
    run_id: str = "",
    provider: str = "",
    model: str = "",
    prompt_version: str = "v1.0",
) -> Path:
    """Write comments to CSV file. Returns the path."""
    run_id = run_id or str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = _build_rows(comments, run_id, provider, model, prompt_version, timestamp)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _STANDARD_COLUMNS + _AUDIT_COLUMNS

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def to_csv_bytes(
    comments: list[GeneratedComment],
    run_id: str = "",
    provider: str = "",
    model: str = "",
    prompt_version: str = "v1.0",
) -> bytes:
    """Return CSV content as bytes (UTF-8 BOM) for browser download."""
    run_id = run_id or str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = _build_rows(comments, run_id, provider, model, prompt_version, timestamp)
    fieldnames = _STANDARD_COLUMNS + _AUDIT_COLUMNS

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    # Prepend UTF-8 BOM
    return ("\ufeff" + buf.getvalue()).encode("utf-8")


def make_filename(product_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = product_name.replace("/", "_").replace("\\", "_")[:20]
    return f"{safe_name}_{ts}.csv"
