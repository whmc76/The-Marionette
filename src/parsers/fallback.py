"""ParserFallback: 弱结构兜底解析器。

将整个文档作为背景上下文，用段落分组生成分类。
"""
from __future__ import annotations

import re
from typing import Optional

from src.models import BriefSpec, CommentCategory
from src.parsers.base import BaseParser, clean_text


class FallbackParser(BaseParser):
    """兜底解析器：当其他解析器置信度低时使用。"""

    name = "fallback"

    def _parse(self, paragraphs: list[str]) -> tuple[Optional[BriefSpec], float, list[str]]:
        warnings = ["使用兜底解析器，解析结果可能不完整，请在确认步骤中手动补充"]

        title = paragraphs[0] if paragraphs else "未知文档"
        # 提取产品名
        product_name = _guess_product_name(title, paragraphs)

        # 将所有段落分组，每组作为一个分类
        long_paras = [p for p in paragraphs if len(p) >= 15]
        # 将长段落直接作为示例评论放入单个"通用"分类
        categories = []
        if long_paras:
            # 尝试按语义分组：前半为正向，后半为引导
            mid = len(long_paras) // 2
            if mid > 0:
                categories.append(CommentCategory(
                    direction="正向",
                    theme="通用正向",
                    sub_themes=[],
                    description="从文档中提取的通用正向内容",
                    personas=[],
                    example_comments=long_paras[:mid][:5],
                ))
            categories.append(CommentCategory(
                direction="引导",
                theme="通用引导",
                sub_themes=[],
                description="从文档中提取的其他内容",
                personas=[],
                example_comments=long_paras[mid:][:5],
            ))

        # 背景 = 所有段落拼接
        background = " ".join(paragraphs[:20])

        # 尝试提取禁用词（短行）
        forbidden = [clean_text(p) for p in paragraphs if 2 <= len(p) <= 20][:20]

        spec = BriefSpec(
            title=title,
            product_name=product_name or "未知产品",
            product_background=background[:500],
            general_rules=["保持自然口语化", "不使用违禁词"],
            forbidden_phrases=forbidden,
            categories=categories,
            positive_ratio=0.5,
            negative_ratio=0.5,
            min_char_length=20,
            platform_targets=[],
        )
        return spec, 0.2, warnings


def _guess_product_name(title: str, paragraphs: list[str]) -> str:
    for text in [title] + paragraphs[:5]:
        m = re.search(r"[\u4e00-\u9fa5]{2,10}", text)
        if m:
            return m.group()
    return ""
