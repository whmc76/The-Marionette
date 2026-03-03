"""Comment generation prompt builder."""
from __future__ import annotations

from src.models import CommentCategory, CommentTask

PROMPT_VERSION = "v1.0"


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

    sub_themes_text = ""
    if cat.sub_themes:
        sub_themes_text = f"\n【可选子主题】（每条评论聚焦其中一个）\n" + "\n".join(f"- {s}" for s in cat.sub_themes)

    persona_text = f"\n【人设扮演】\n你现在是：{task.persona}" if task.persona else ""

    examples_text = ""
    if cat.example_comments:
        examples_text = "\n【参考示例】（风格参考，不要直接复制）\n" + "\n".join(
            f"- {e}" for e in cat.example_comments[:3]
        )

    avoid_text = ""
    if already_generated:
        # 只展示最近10条避免prompt过长
        recent = already_generated[-10:]
        avoid_text = "\n【已生成评论片段（避免重复）】\n" + "\n".join(
            f"- {c[:50]}..." if len(c) > 50 else f"- {c}"
            for c in recent
        )

    desc_text = f"\n【分类描述】\n{cat.description}" if cat.description else ""

    return f"""请生成 {task.target_count} 条「{direction_desc}」类型的用户评论。

【主题】{cat.theme}
【评论方向】{direction_desc}{sub_themes_text}{desc_text}{persona_text}{examples_text}{avoid_text}

要求：
- 生成恰好 {task.target_count} 条，编号 1. 到 {task.target_count}.
- 每条独立一行
- 风格差异化，避免套路化表达
- 不同条评论之间角度要有变化
- 禁止直接复制示例或已生成内容
"""
