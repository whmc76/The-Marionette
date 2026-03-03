"""LLM-based brief parser — two-phase extraction with per-field progress callbacks."""
from __future__ import annotations

import json
import re
from typing import Callable, Optional

from src.llm.client import LLMClient
from src.models import BriefSpec, CommentCategory, ParseReport
from src.parsers.base import BaseParser, load_paragraphs
from src.presets import get_preset, IndustryPreset

# Callback type: on_step(field_name, result_summary)
StepCallback = Callable[[str, str], None]
# Token callback: on_token(chunk) — None chunk signals "clear / new phase"
TokenCallback = Callable[[Optional[str]], None]

_SYSTEM = """\
你是一名营销文档结构化提取专家，专门解析评论运营 brief 文档。
只返回 JSON，不要有任何额外说明、注释或 markdown 代码块标记。
"""

# ── Phase 1: Basic info ───────────────────────────────────────────

_BASIC_PROMPT = """\
从以下 brief 文档中提取基础信息，只返回如下 JSON，没有信息用空字符串/空列表/默认值：

{{
  "title": "文档标题",
  "product_name": "产品名称（如：{product_example}）",
  "product_background": "产品背景与定位，100字以内",
  "general_rules": ["写作规则1", "写作规则2"],
  "forbidden_phrases": ["禁用词1", "禁用词2"],
  "positive_ratio": 0.5,
  "min_char_length": 20,
  "platform_targets": ["微博", "小红书"]
}}

注意：
- forbidden_phrases 提取所有禁止使用的词语、短语、广告用语
- general_rules 提取通用写作要求、语气要求等
- positive_ratio 根据文档中正向/反击/引导分类的数量比例估算（0~1）
- 以上示例仅作格式参考，只提取文档中实际存在的信息，不要虚构文档中不存在的内容。

=== 文档内容 ===
{document}
"""

# ── Phase 2: Categories ───────────────────────────────────────────

_CATEGORIES_PROMPT = """\
从以下 brief 文档中提取所有评论分类，只返回 JSON 数组，每个元素结构如下：

[
  {{
    "direction": "正向",
    "theme": "主题名称（如：{theme_example}）",
    "sub_themes": ["子主题1", "子主题2"],
    "description": "该分类的写作指导与角度说明",
    "personas": ["人设1（如：{persona_example}）", "人设2"],
    "example_comments": ["示例评论原文（尽量保留文档原文）"]
  }}
]

注意：
- direction 只能是 "正向"、"反击"、"引导" 三者之一
- 必须覆盖文档中所有评论分类，不要遗漏
- example_comments 尽量保留原文，每条示例单独一个字符串
- personas 提取文档中的账号人设、发帖身份描述
- 以上示例仅作格式参考，只提取文档中实际存在的信息，不要虚构文档中不存在的内容。

=== 文档内容 ===
{document}
"""


class LLMParser(BaseParser):
    """Two-phase LLM extraction with per-field progress callbacks."""

    name = "llm"

    def __init__(
        self,
        client: LLMClient,
        on_step: Optional[StepCallback] = None,
        on_token: Optional[TokenCallback] = None,
        industry: str = "general",
    ) -> None:
        self._client = client
        self._on_step = on_step or (lambda field, result: None)
        self._on_token = on_token
        self._preset: IndustryPreset = get_preset(industry)

    # BaseParser calls this
    def _parse(self, paragraphs: list[str]) -> tuple[Optional[BriefSpec], float, list[str]]:
        warnings: list[str] = []

        doc_text = "\n".join(paragraphs)

        # Dynamic context window: measure doc length, compute minimum safe num_ctx.
        # Hard cap at ~260k chars (131072 ctx ≈ 130k tokens, generous buffer).
        if len(doc_text) > 260_000:
            doc_text = doc_text[:260_000]
            warnings.append("文档超长（>260000字），已截取前260000字符。如解析不完整请精简文档。")

        required_ctx = _required_ctx(doc_text)
        _extra = {"num_ctx": required_ctx, "think": False}
        ctx_label = f"动态扩展 → {required_ctx:,}" if required_ctx > 8192 else str(required_ctx)
        self._on_step("上下文窗口", f"{ctx_label} tokens（文档 {len(doc_text):,} 字）")

        # ── Phase 1: Basic info ───────────────────────────────────
        basic = self._extract_basic(doc_text, warnings, _extra)

        # ── Phase 2: Categories ───────────────────────────────────
        categories = self._extract_categories(doc_text, warnings, _extra)

        if basic is None and not categories:
            return None, 0.0, warnings

        # Merge into BriefSpec
        base = basic or {}
        try:
            pos_ratio = float(base.get("positive_ratio") or 0.5)
        except (ValueError, TypeError):
            pos_ratio = 0.5
        pos_ratio = max(0.0, min(1.0, pos_ratio))

        try:
            min_char_length = int(base.get("min_char_length") or 20)
        except (ValueError, TypeError):
            min_char_length = 20

        spec = BriefSpec(
            title=str(base.get("title", "")).strip(),
            product_name=str(base.get("product_name", "")).strip(),
            product_background=str(base.get("product_background", "")).strip(),
            general_rules=[str(r).strip() for r in base.get("general_rules", []) if r],
            forbidden_phrases=[str(p).strip() for p in base.get("forbidden_phrases", []) if p],
            categories=categories,
            positive_ratio=pos_ratio,
            negative_ratio=round(1.0 - pos_ratio, 2),
            min_char_length=min_char_length,
            platform_targets=[str(p).strip() for p in base.get("platform_targets", []) if p],
            industry=self._preset.key,
        )

        confidence = 0.7
        if spec.categories:
            confidence += 0.15
        if spec.product_name:
            confidence += 0.10
        if spec.general_rules or spec.forbidden_phrases:
            confidence += 0.05

        if not spec.categories:
            warnings.append("未提取到任何评论分类，请在下一步手动添加")
        if not spec.product_name:
            warnings.append("未识别到产品名称")

        return spec, min(confidence, 1.0), warnings

    # ── Phase helpers ─────────────────────────────────────────────

    def _extract_basic(self, doc_text: str, warnings: list[str], extra_options: Optional[dict] = None) -> Optional[dict]:
        cb = self._on_step

        cb("产品名称", "…")
        cb("产品背景", "…")
        cb("通用规则", "…")
        cb("禁用词", "…")
        cb("平台 / 比例", "…")

        preset = self._preset
        product_example = "、".join(preset.product_examples)
        prompt = _BASIC_PROMPT.format(product_example=product_example, document=doc_text)

        if self._on_token:
            self._on_token(None)  # signal: clear stream buffer / new phase
        try:
            if self._on_token:
                resp = self._client.stream_chat(
                    system=_SYSTEM,
                    user=prompt,
                    temperature=0.1,
                    on_chunk=self._on_token,
                    extra_options=extra_options,
                )
            else:
                resp = self._client.chat(
                    system=_SYSTEM,
                    user=prompt,
                    temperature=0.1,
                    extra_options=extra_options,
                )
            data = _extract_json(resp.content)
        except Exception as exc:
            warnings.append(f"基础信息提取失败: {exc}")
            cb("产品名称", "❌ 失败")
            return None

        # Report each field
        cb("产品名称", data.get("product_name") or "（未识别）")
        cb("产品背景", f"{len(data.get('product_background', ''))} 字")
        cb("通用规则", f"{len(data.get('general_rules', []))} 条")
        cb("禁用词", f"{len(data.get('forbidden_phrases', []))} 条")
        platforms = data.get("platform_targets", [])
        cb("平台 / 比例", f"{', '.join(platforms) or '未识别'} · 正向 {data.get('positive_ratio', 0.5):.0%}")
        return data

    def _extract_categories(self, doc_text: str, warnings: list[str], extra_options: Optional[dict] = None) -> list[CommentCategory]:
        cb = self._on_step
        cb("评论分类", "…")

        preset = self._preset
        theme_example = "、".join(preset.theme_examples)
        persona_example = "、".join(preset.persona_examples)
        prompt = _CATEGORIES_PROMPT.format(
            theme_example=theme_example,
            persona_example=persona_example,
            document=doc_text,
        )

        if self._on_token:
            self._on_token(None)  # signal: clear stream buffer / new phase
        try:
            if self._on_token:
                resp = self._client.stream_chat(
                    system=_SYSTEM,
                    user=prompt,
                    temperature=0.1,
                    on_chunk=self._on_token,
                    extra_options=extra_options,
                )
            else:
                resp = self._client.chat(
                    system=_SYSTEM,
                    user=prompt,
                    temperature=0.1,
                    extra_options=extra_options,
                )
            raw = resp.content.strip()
            # Response is a JSON array
            data = _extract_json_array(raw)
        except Exception as exc:
            warnings.append(f"分类提取失败: {exc}")
            cb("评论分类", "❌ 失败")
            return []

        categories: list[CommentCategory] = []
        persona_set: set[str] = set()
        example_total = 0

        for cat_data in data:
            direction = str(cat_data.get("direction", "正向")).strip()
            if direction not in ("正向", "反击", "引导"):
                direction = "正向"
            personas = [str(p).strip() for p in cat_data.get("personas", []) if p]
            examples = [str(e).strip() for e in cat_data.get("example_comments", []) if e]
            persona_set.update(personas)
            example_total += len(examples)

            theme = str(cat_data.get("theme", "未命名")).strip()
            cb(f"  [{direction}] {theme}",
               f"{len(personas)} 种人设 · {len(examples)} 条示例")

            categories.append(CommentCategory(
                direction=direction,
                theme=theme,
                sub_themes=[str(s).strip() for s in cat_data.get("sub_themes", []) if s],
                description=str(cat_data.get("description", "")).strip(),
                personas=personas,
                example_comments=examples,
            ))

        cb("评论分类", f"共 {len(categories)} 个 · {len(persona_set)} 种人设 · {example_total} 条示例")
        return categories


# ── JSON helpers ──────────────────────────────────────────────────

def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


def _extract_json(text: str) -> dict:
    """Extract first JSON object from LLM response."""
    text = _strip_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("No JSON object found in response")


def _required_ctx(doc_text: str) -> int:
    """Dynamically estimate minimum num_ctx for parsing this document.

    Chinese text ≈ 2 chars/token (conservative estimate).
    Adds 1100 tokens overhead for prompt templates + expected output.
    Rounds up to the nearest standard Ollama context size.
    """
    needed = len(doc_text) // 2 + 1100
    for size in (4096, 8192, 16384, 32768, 65536, 131072):
        if needed <= size:
            return size
    return 131072


def _extract_json_array(text: str) -> list:
    """Extract first JSON array of objects from LLM response."""
    text = _strip_fences(text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        # Wrapped object: find the first value that is a list of dicts
        if isinstance(result, dict):
            for v in result.values():
                if isinstance(v, list) and (not v or isinstance(v[0], dict)):
                    return v
    except json.JSONDecodeError:
        pass
    # Find first [...] block in raw text
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "[":
            if start is None:
                start = i
            depth += 1
        elif ch == "]" and depth > 0:
            depth -= 1
            if depth == 0:
                return json.loads(text[start: i + 1])
    raise ValueError("No JSON array found in response")
