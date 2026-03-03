"""Comment generation prompt builder."""
from __future__ import annotations

import re
from src.models import CommentCategory, CommentTask

PROMPT_VERSION = "v1.1"

# ── Diversity pools ───────────────────────────────────────────────

# Writing perspectives — cycled across comment slots
_PERSPECTIVES = [
    "亲身体验（用第一人称讲自己的使用感受）",
    "数据/事实角度（引用具体数字、参数或对比数据）",
    "场景描述（描述一个具体的使用场景或情境）",
    "推荐/种草口吻（像在向朋友推荐）",
    "转折式（先提出疑虑或质疑，再给出正面判断）",
    "对比视角（和竞品、老款或市场同类产品对比）",
    "情感表达（以情绪和感受为主，少用数据）",
    "路人评论（旁观者视角，偶然看到/听说）",
    "追问/好奇（带疑问句，表达期待或想了解更多）",
    "幽默/调侃（轻松口吻，带一点玩笑感）",
]

# Sentence forms — cycled across comment slots
_FORMS = [
    "陈述句（平铺直叙，说清楚事实）",
    "感叹句（语气强烈，带感叹号）",
    "疑问/反问句（以问号结尾或包含反问）",
    "转折句（先说but/虽然，再转入正面）",
    "对话体（像和朋友说话，口语感强）",
]

# Length targets — cycled across comment slots
_LENGTHS = [
    "短评：20-40字",
    "中评：40-70字",
    "长评：70-120字",
]


def _slot_specs(n: int, sub_themes: list[str]) -> list[str]:
    """Generate per-slot diversity specs for n comment slots."""
    specs = []
    for i in range(n):
        perspective = _PERSPECTIVES[i % len(_PERSPECTIVES)]
        form = _FORMS[i % len(_FORMS)]
        length = _LENGTHS[i % len(_LENGTHS)]
        sub = f"，子主题聚焦「{sub_themes[i % len(sub_themes)]}」" if sub_themes else ""
        specs.append(
            f"{i + 1}. 【视角】{perspective}｜【形式】{form}｜【长度】{length}{sub}"
        )
    return specs


def _forbidden_openers(already_generated: list[str], top_n: int = 8) -> list[str]:
    """Extract the most common sentence-opening 2-gram patterns to blacklist."""
    openers: dict[str, int] = {}
    for text in already_generated:
        text = text.strip()
        if len(text) >= 2:
            key = text[:2]
            openers[key] = openers.get(key, 0) + 1
    # Blacklist any opener already used at least once
    frequent = [k for k, v in openers.items() if v >= 1]
    frequent.sort(key=lambda k: -openers[k])
    return frequent[:top_n]


def build_comment_prompt(
    task: CommentTask,
    already_generated: list[str],
) -> str:
    cat = task.category
    direction_desc = {
        "正向": "正向好评（真实用户的积极体验分享）",
        "反击": "反击/应对（针对竞品或负面言论的自然反驳）",
        "引导": "舆论引导（带动话题讨论，引导正向舆论）",
    }.get(cat.direction, cat.direction)

    persona_text = f"\n【人设扮演】\n你现在是：{task.persona}" if task.persona else ""

    examples_text = ""
    if cat.example_comments:
        examples_text = "\n【参考示例】（仅供风格参考，禁止复制原文）\n" + "\n".join(
            f"- {e}" for e in cat.example_comments[:3]
        )

    desc_text = f"\n【分类描述】\n{cat.description}" if cat.description else ""

    # Per-slot diversity specs
    slot_lines = _slot_specs(task.target_count, cat.sub_themes)
    slots_text = "\n".join(slot_lines)

    # Forbidden openers from already-generated content
    forbidden_openers = _forbidden_openers(already_generated)
    opener_warn = ""
    if forbidden_openers:
        opener_warn = f"\n【禁止使用的开头】（以下字/词开头的已生成太多，换别的开头）\n" + "、".join(
            f"「{o}」" for o in forbidden_openers
        )

    # Recent samples to avoid near-duplicate content
    avoid_text = ""
    if already_generated:
        recent = already_generated[-12:]
        avoid_text = "\n【已生成内容（内容和句式不得相似）】\n" + "\n".join(
            f"- {c[:60]}…" if len(c) > 60 else f"- {c}"
            for c in recent
        )

    return f"""请生成 {task.target_count} 条「{direction_desc}」类型的用户评论。

【主题】{cat.theme}
【评论方向】{direction_desc}{desc_text}{persona_text}{examples_text}{avoid_text}{opener_warn}

【每条评论的差异化要求】（必须严格按照每条的指定视角、形式、长度来写）
{slots_text}

输出规则：
- 严格按编号 1. 到 {task.target_count}. 输出，每条独立一行
- 每条必须符合对应编号的【视角】【形式】【长度】要求
- 禁止不同编号之间使用相同的开头词或相同的句式结构
- 禁止复制示例原文或已生成内容
- 不加任何额外说明或标题
"""
