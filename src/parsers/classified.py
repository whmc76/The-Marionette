"""ParserA: 分类规则型 brief（如分类规则型营销 brief）。

特征：包含"正向-"/"反击-"分类标记、人设列表、产品数据块。
"""
from __future__ import annotations

import re
from typing import Optional

from src.models import BriefSpec, CommentCategory
from src.parsers.base import BaseParser, clean_text


# ── 识别标记 ──────────────────────────────────────────────────
_POSITIVE_MARKERS = re.compile(r"正向|好评|优势|支持|推荐")
_NEGATIVE_MARKERS = re.compile(r"反击|应对|回怼|负面|差评|防御")
_GUIDE_MARKERS = re.compile(r"引导|转化|话题|舆论")
_PERSONA_SECTION = re.compile(r"人设|身份|角色|账号")
_FORBIDDEN_SECTION = re.compile(r"禁止|不得|禁用|避免|不要")
_PRODUCT_SECTION = re.compile(r"产品.*?(背景|介绍|亮点|特点|数据|参数|配置)")
_RULE_SECTION = re.compile(r"总体.*?规则|写作.*?原则|通用.*?要求|注意.*?事项")
_PLATFORM_SECTION = re.compile(r"平台|渠道|微博|小红书|抖音|B站|知乎")


class ClassifiedParser(BaseParser):
    """解析"分类规则型"brief（分类规则型 brief）。"""

    name = "classified"

    def _parse(self, paragraphs: list[str]) -> tuple[Optional[BriefSpec], float, list[str]]:
        warnings: list[str] = []
        signals = 0  # confidence signals

        title = paragraphs[0] if paragraphs else ""
        product_name = ""
        product_background_lines: list[str] = []
        general_rules: list[str] = []
        forbidden_phrases: list[str] = []
        platform_targets: list[str] = []
        categories: list[CommentCategory] = []
        global_personas: list[str] = []

        # ── 第一遍：收集产品/规则/人设/平台 ────────────────────
        i = 0
        while i < len(paragraphs):
            p = paragraphs[i]

            if _PRODUCT_SECTION.search(p):
                signals += 1
                # 收集后续段落作为背景
                j = i + 1
                while j < len(paragraphs) and not _any_section(paragraphs[j]):
                    product_background_lines.append(paragraphs[j])
                    j += 1
                if not product_name:
                    # 尝试从标题或第一行提取产品名
                    product_name = _extract_product_name(title, paragraphs)

            elif _RULE_SECTION.search(p):
                signals += 1
                j = i + 1
                while j < len(paragraphs) and not _any_section(paragraphs[j]):
                    rule = clean_text(paragraphs[j])
                    if rule and len(rule) > 3:
                        general_rules.append(rule)
                    j += 1

            elif _FORBIDDEN_SECTION.search(p):
                signals += 1
                j = i + 1
                while j < len(paragraphs) and not _any_section(paragraphs[j]):
                    phrase = clean_text(paragraphs[j])
                    if phrase and len(phrase) > 1:
                        # 拆分逗号/顿号分隔的多个词
                        forbidden_phrases.extend(
                            w.strip() for w in re.split(r"[，,、；;]", phrase) if w.strip()
                        )
                    j += 1

            elif _PERSONA_SECTION.search(p):
                signals += 1
                j = i + 1
                while j < len(paragraphs) and not _any_section(paragraphs[j]):
                    persona = clean_text(paragraphs[j])
                    if persona and len(persona) > 1:
                        global_personas.append(persona)
                    j += 1

            elif _PLATFORM_SECTION.search(p):
                signals += 1
                j = i + 1
                while j < len(paragraphs) and not _any_section(paragraphs[j]):
                    pt = clean_text(paragraphs[j])
                    if pt:
                        platform_targets.extend(
                            w.strip() for w in re.split(r"[，,、\s]+", pt) if w.strip()
                        )
                    j += 1
            i += 1

        # ── 第二遍：解析分类块 ──────────────────────────────────
        categories = _extract_categories(paragraphs, global_personas)
        if categories:
            signals += 2

        # ── 置信度 ──────────────────────────────────────────────
        confidence = min(1.0, signals / 7.0)

        # 低信号时警告
        if len(categories) == 0:
            warnings.append("未识别到任何分类，请检查 brief 格式")
        if not general_rules:
            warnings.append("未识别到通用规则段落")
        if not product_name:
            product_name = _extract_product_name(title, paragraphs)
            if not product_name:
                product_name = "未知产品"
                warnings.append("未识别到产品名称")

        # 正负比例
        pos = sum(1 for c in categories if "正向" in c.direction)
        neg = sum(1 for c in categories if "反击" in c.direction or "引导" in c.direction)
        total = len(categories) or 1
        pos_ratio = round(pos / total, 2)
        neg_ratio = round(1 - pos_ratio, 2)

        spec = BriefSpec(
            title=title,
            product_name=product_name,
            product_background=" ".join(product_background_lines[:10]),
            general_rules=general_rules[:30],
            forbidden_phrases=list(set(forbidden_phrases))[:50],
            categories=categories,
            positive_ratio=pos_ratio,
            negative_ratio=neg_ratio,
            min_char_length=20,
            platform_targets=list(set(platform_targets))[:10],
        )
        return spec, confidence, warnings


# ── 工具函数 ────────────────────────────────────────────────────

_SECTION_HEADERS = re.compile(
    r"产品|规则|原则|要求|事项|人设|身份|角色|账号|平台|渠道|禁止|不得|正向|反击|引导"
)


def _any_section(text: str) -> bool:
    return bool(_SECTION_HEADERS.search(text)) and len(text) < 60


def _extract_product_name(title: str, paragraphs: list[str]) -> str:
    # 尝试从标题/前几行提取产品名（中文产品名通常2-8个汉字）
    candidates = [title] + paragraphs[:5]
    for c in candidates:
        m = re.search(r"[\u4e00-\u9fa5]{2,8}[\u4e00-\u9fa5]{0,4}", c)
        if m:
            return m.group()
    return ""


def _extract_categories(paragraphs: list[str], global_personas: list[str]) -> list[CommentCategory]:
    """从段落中提取正向/反击分类块。"""
    categories: list[CommentCategory] = []
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        direction = None

        if _POSITIVE_MARKERS.search(p) and len(p) < 80:
            direction = "正向"
        elif _NEGATIVE_MARKERS.search(p) and len(p) < 80:
            direction = "反击"
        elif _GUIDE_MARKERS.search(p) and len(p) < 80:
            direction = "引导"

        if direction:
            # 主题名：去掉方向关键词后剩余文本，或直接用段落
            theme = re.sub(r"正向|反击|引导|好评|负面|话题|舆论|-|：|:", "", p).strip() or p
            theme = clean_text(theme) or f"{direction}类"

            sub_themes: list[str] = []
            description_lines: list[str] = []
            example_comments: list[str] = []
            local_personas: list[str] = list(global_personas)

            j = i + 1
            while j < len(paragraphs):
                nxt = paragraphs[j]
                if _is_new_category(nxt):
                    break
                # 子主题（短句，可能带序号）
                if re.match(r"^[①-⑳\d一二三四五六七八九十]+[\.、．]", nxt) and len(nxt) < 60:
                    sub_themes.append(clean_text(re.sub(r"^[①-⑳\d一二三四五六七八九十]+[\.、．]", "", nxt)))
                # 示例评论（较长段落，通常包含引号或感叹号）
                elif len(nxt) >= 30 and ('"' in nxt or '"' in nxt or '！' in nxt or '~' in nxt):
                    example_comments.append(nxt)
                # 人设覆盖
                elif _PERSONA_SECTION.search(nxt):
                    k = j + 1
                    local_personas = []
                    while k < len(paragraphs) and not _is_new_category(paragraphs[k]):
                        lp = clean_text(paragraphs[k])
                        if lp:
                            local_personas.append(lp)
                        k += 1
                    j = k
                    continue
                else:
                    description_lines.append(nxt)
                j += 1

            categories.append(CommentCategory(
                direction=direction,
                theme=theme,
                sub_themes=sub_themes[:10],
                description=" ".join(description_lines[:5]),
                personas=local_personas[:10],
                example_comments=example_comments[:5],
            ))
            i = j
        else:
            i += 1

    return categories


def _is_new_category(text: str) -> bool:
    return bool(
        (_POSITIVE_MARKERS.search(text) or _NEGATIVE_MARKERS.search(text) or _GUIDE_MARKERS.search(text))
        and len(text) < 80
    ) or bool(_PRODUCT_SECTION.search(text) or _RULE_SECTION.search(text) or _FORBIDDEN_SECTION.search(text))
