"""ParserB: 话术库型 brief（如话术库型营销 brief）。

特征：包含预写评论模板、"评论方向"/"话术"/"维护建议"标记。
"""
from __future__ import annotations

import re
from typing import Optional

from src.models import BriefSpec, CommentCategory
from src.parsers.base import BaseParser, clean_text


_DIRECTION_MARKERS = re.compile(r"评论方向|评论类型|方向|类别|场景")
_SCRIPT_MARKERS = re.compile(r"话术|评论模板|示例|评论内容|文案|参考评论")
_MAINTAIN_MARKERS = re.compile(r"维护.*?建议|运营.*?建议|平台.*?策略|注意.*?事项|发布.*?规则")
_POSITIVE_WORDS = re.compile(r"正向|好评|推荐|优势|亮点|支持")
_NEGATIVE_WORDS = re.compile(r"引导|反击|应对|话题|舆论|维权|竞品")
_PRODUCT_SECTION = re.compile(r"产品.*?(背景|介绍|概述)|品牌")
_FORBIDDEN_SECTION = re.compile(r"禁止|不得|禁用|敏感词|避免")
_PLATFORM_SECTION = re.compile(r"平台|渠道|微博|小红书|抖音|B站|知乎|微信")


class ScriptLibParser(BaseParser):
    """解析"话术库+舆情引导型"brief（话术库 brief）。"""

    name = "script_lib"

    def _parse(self, paragraphs: list[str]) -> tuple[Optional[BriefSpec], float, list[str]]:
        warnings: list[str] = []
        signals = 0

        title = paragraphs[0] if paragraphs else ""
        product_name = _extract_product_name(title, paragraphs)
        product_background_lines: list[str] = []
        general_rules: list[str] = []
        forbidden_phrases: list[str] = []
        platform_targets: list[str] = []

        # ── 第一遍：收集产品/维护规则/禁用词/平台 ──────────────
        i = 0
        while i < len(paragraphs):
            p = paragraphs[i]

            if _PRODUCT_SECTION.search(p):
                signals += 1
                j = i + 1
                while j < len(paragraphs) and not _is_section_header(paragraphs[j]):
                    product_background_lines.append(paragraphs[j])
                    j += 1

            elif _MAINTAIN_MARKERS.search(p):
                signals += 1
                j = i + 1
                while j < len(paragraphs) and not _is_section_header(paragraphs[j]):
                    rule = clean_text(paragraphs[j])
                    if rule and len(rule) > 3:
                        general_rules.append(rule)
                    j += 1

            elif _FORBIDDEN_SECTION.search(p):
                signals += 1
                j = i + 1
                while j < len(paragraphs) and not _is_section_header(paragraphs[j]):
                    phrase = clean_text(paragraphs[j])
                    if phrase:
                        forbidden_phrases.extend(
                            w.strip() for w in re.split(r"[，,、；;\s]+", phrase) if w.strip()
                        )
                    j += 1

            elif _PLATFORM_SECTION.search(p) and len(p) < 50:
                signals += 1
                platforms = re.findall(r"微博|小红书|抖音|B站|知乎|微信|APP|论坛", p)
                platform_targets.extend(platforms)
                j = i + 1
                while j < len(paragraphs) and not _is_section_header(paragraphs[j]):
                    more = re.findall(r"微博|小红书|抖音|B站|知乎|微信|APP|论坛", paragraphs[j])
                    platform_targets.extend(more)
                    j += 1
            i += 1

        # ── 第二遍：提取话术分类块 ──────────────────────────────
        categories = _extract_script_categories(paragraphs)
        if categories:
            signals += 3

        confidence = min(1.0, signals / 7.0)

        if len(categories) == 0:
            warnings.append("未识别到话术分类块，请检查 brief 格式")
        if not product_name:
            product_name = "未知产品"
            warnings.append("未识别到产品名称")

        pos = sum(1 for c in categories if "正向" in c.direction)
        total = len(categories) or 1
        pos_ratio = round(pos / total, 2)

        spec = BriefSpec(
            title=title,
            product_name=product_name,
            product_background=" ".join(product_background_lines[:10]),
            general_rules=general_rules[:30],
            forbidden_phrases=list(set(forbidden_phrases))[:50],
            categories=categories,
            positive_ratio=pos_ratio,
            negative_ratio=round(1 - pos_ratio, 2),
            min_char_length=20,
            platform_targets=list(set(platform_targets))[:10],
        )
        return spec, confidence, warnings


# ── 工具函数 ────────────────────────────────────────────────────

_SECTION_PATTERNS = re.compile(
    r"产品|维护|平台|规则|禁止|不得|评论方向|话术|示例|方向|类别|场景|运营|背景|介绍"
)


def _is_section_header(text: str) -> bool:
    return bool(_SECTION_PATTERNS.search(text)) and len(text) < 60


def _extract_product_name(title: str, paragraphs: list[str]) -> str:
    candidates = [title] + paragraphs[:8]
    for c in candidates:
        m = re.search(r"[\u4e00-\u9fa5]{2,8}(?:\s*[\u4e00-\u9fa5A-Za-z0-9]{1,10})?", c)
        if m:
            return m.group()
    return ""


def _extract_script_categories(paragraphs: list[str]) -> list[CommentCategory]:
    """提取话术库中的评论分类。"""
    categories: list[CommentCategory] = []
    i = 0
    current_direction = None
    current_theme = ""

    while i < len(paragraphs):
        p = paragraphs[i]

        # 检测方向头部
        if _DIRECTION_MARKERS.search(p) and len(p) < 60:
            current_direction = _infer_direction(p)
            current_theme = clean_text(
                re.sub(r"评论方向|评论类型|方向|类别|场景|：|:", "", p)
            ) or current_direction or "通用"
            i += 1
            continue

        # 检测话术块头部
        if _SCRIPT_MARKERS.search(p) and len(p) < 80:
            signals_direction = current_direction or _infer_direction(p) or "正向"
            theme = current_theme or clean_text(
                re.sub(r"话术|评论模板|示例|评论内容|文案|参考评论|：|:", "", p)
            ) or "通用话术"

            examples: list[str] = []
            sub_themes: list[str] = []
            desc_lines: list[str] = []

            j = i + 1
            while j < len(paragraphs):
                nxt = paragraphs[j]
                if _SCRIPT_MARKERS.search(nxt) or _DIRECTION_MARKERS.search(nxt) or _is_section_header(nxt):
                    break
                # 识别示例评论：长度适中且有自然语言特征
                if 15 <= len(nxt) <= 200 and not _is_section_header(nxt):
                    examples.append(nxt)
                elif len(nxt) < 60 and re.match(r"^[①-⑳\d一二三四五六七八九十]+", nxt):
                    sub_themes.append(clean_text(re.sub(r"^[①-⑳\d一二三四五六七八九十]+[\.、．]?", "", nxt)))
                else:
                    desc_lines.append(nxt)
                j += 1

            if examples or sub_themes:
                categories.append(CommentCategory(
                    direction=signals_direction,
                    theme=theme,
                    sub_themes=sub_themes[:10],
                    description=" ".join(desc_lines[:3]),
                    personas=[],  # 话术库型通常无固定人设
                    example_comments=examples[:8],
                ))
            i = j
            continue

        i += 1

    return categories


def _infer_direction(text: str) -> str:
    if _POSITIVE_WORDS.search(text):
        return "正向"
    if _NEGATIVE_WORDS.search(text):
        return "引导"
    return "正向"


