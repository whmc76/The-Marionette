"""Unified Pydantic data contracts for The Marionette."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Optional


class CommentCategory(BaseModel):
    direction: str          # "正向" / "反击" / "引导"
    theme: str              # 主题名
    sub_themes: list[str] = Field(default_factory=list)
    description: str = ""   # 指导描述
    personas: list[str] = Field(default_factory=list)
    example_comments: list[str] = Field(default_factory=list)  # few-shot


class BriefSpec(BaseModel):
    title: str = ""
    product_name: str = ""
    product_background: str = ""
    general_rules: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    categories: list[CommentCategory] = Field(default_factory=list)
    positive_ratio: float = 0.5
    negative_ratio: float = 0.5
    min_char_length: int = 20
    platform_targets: list[str] = Field(default_factory=list)
    industry: str = "general"   # 行业预设 key，全链路唯一真值源

    @field_validator("positive_ratio", "negative_ratio", mode="before")
    @classmethod
    def clamp_ratio(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class CommentTask(BaseModel):
    task_id: str
    category: CommentCategory
    persona: str
    target_count: int
    batch_index: int


class GeneratedComment(BaseModel):
    task_id: str
    text: str
    category_direction: str
    theme: str
    sub_theme: str = ""
    persona: str
    char_count: int
    validation_status: str = "pending"  # "pass" / "hard_fail" / "soft_flag"
    errors: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    total: int
    passed: int
    hard_failed: int
    soft_flagged: int
    duplicate_pairs: list[tuple[int, int]] = Field(default_factory=list)


class ParseReport(BaseModel):
    parser_name: str
    confidence: float          # 0.0 - 1.0
    category_count: int
    persona_count: int
    forbidden_phrase_count: int
    example_count: int
    warnings: list[str] = Field(default_factory=list)
    spec: Optional[BriefSpec] = None
