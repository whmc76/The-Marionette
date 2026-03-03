"""Industry preset definitions for The Marionette.

Each preset provides industry-specific defaults for UI placeholders,
LLM prompt examples, and the system prompt role description.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IndustryPreset:
    key: str                      # canonical key, e.g. "automotive"
    label: str                    # Chinese label, e.g. "汽车"
    icon: str                     # emoji icon
    role_desc: str                # injected into system prompt role line
    product_examples: list[str]   # UI placeholder + LLM prompt examples
    theme_examples: list[str]
    persona_examples: list[str]
    platform_defaults: list[str]
    lexicon: list[str]            # domain keywords for optional parser hints


PRESETS: list[IndustryPreset] = [
    IndustryPreset(
        key="automotive",
        label="汽车",
        icon="🚗",
        role_desc="汽车品牌",
        product_examples=["岚图梦想家", "启境"],
        theme_examples=["续航表现", "空间体验"],
        persona_examples=["宝妈", "科技博主"],
        platform_defaults=["微博", "小红书", "抖音"],
        lexicon=["车型", "续航", "动力", "新能源", "电动", "里程", "充电"],
    ),
    IndustryPreset(
        key="beauty",
        label="美妆护肤",
        icon="💄",
        role_desc="美妆品牌",
        product_examples=["某品牌精华液", "保湿面霜"],
        theme_examples=["保湿效果", "成分安全"],
        persona_examples=["敏感肌用户", "护肤博主"],
        platform_defaults=["小红书", "微博", "抖音"],
        lexicon=["面霜", "精华", "护肤", "成分", "肤质", "保湿", "美白"],
    ),
    IndustryPreset(
        key="tech",
        label="3C数码",
        icon="📱",
        role_desc="数码品牌",
        product_examples=["某品牌旗舰手机", "真无线耳机"],
        theme_examples=["拍照体验", "续航表现"],
        persona_examples=["数码发烧友", "学生党"],
        platform_defaults=["微博", "B站", "知乎"],
        lexicon=["手机", "耳机", "性能", "拍照", "芯片", "屏幕", "电池"],
    ),
    IndustryPreset(
        key="food",
        label="餐饮美食",
        icon="🍜",
        role_desc="餐饮品牌",
        product_examples=["某连锁火锅", "网红奶茶"],
        theme_examples=["口味体验", "门店环境"],
        persona_examples=["美食博主", "上班族"],
        platform_defaults=["大众点评", "小红书", "抖音"],
        lexicon=["门店", "菜品", "口味", "外卖", "口感", "食材", "服务"],
    ),
    IndustryPreset(
        key="education",
        label="教育培训",
        icon="📚",
        role_desc="教育品牌",
        product_examples=["某在线课程平台", "英语培训班"],
        theme_examples=["课程质量", "老师专业度"],
        persona_examples=["学生家长", "在职学员"],
        platform_defaults=["微博", "知乎", "小红书"],
        lexicon=["课程", "老师", "学习", "培训", "考试", "辅导", "效果"],
    ),
    IndustryPreset(
        key="baby",
        label="母婴",
        icon="🍼",
        role_desc="母婴品牌",
        product_examples=["某品牌奶粉", "纸尿裤"],
        theme_examples=["安全成分", "宝宝适用性"],
        persona_examples=["新手妈妈", "育儿达人"],
        platform_defaults=["小红书", "微博", "抖音"],
        lexicon=["奶粉", "纸尿裤", "辅食", "宝宝", "安全", "婴儿", "喂养"],
    ),
    IndustryPreset(
        key="general",
        label="通用",
        icon="🏷️",
        role_desc="品牌",
        product_examples=["某品牌产品"],
        theme_examples=["产品体验", "品牌服务"],
        persona_examples=["真实用户", "普通消费者"],
        platform_defaults=["微博", "小红书"],
        lexicon=["产品", "品牌"],
    ),
]

_PRESET_MAP: dict[str, IndustryPreset] = {p.key: p for p in PRESETS}


def get_preset(key: str) -> IndustryPreset:
    """Return the preset for the given key, falling back to 'general'."""
    return _PRESET_MAP.get(key, _PRESET_MAP["general"])
