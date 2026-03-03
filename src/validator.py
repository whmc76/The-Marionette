"""Layered validation: hard rules + soft rules."""
from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz

from src.models import BriefSpec, GeneratedComment, ValidationResult


# ── Hard rules ───────────────────────────────────────────────────


def _check_length(comment: GeneratedComment, min_len: int) -> Optional[str]:
    if comment.char_count < min_len:
        return f"字数不足 {comment.char_count}/{min_len}"
    return None


def _check_forbidden(comment: GeneratedComment, forbidden: list[str]) -> Optional[str]:
    for phrase in forbidden:
        if phrase and phrase in comment.text:
            return f"含禁用词: 「{phrase}」"
    return None


_TRAILING_PUNCT = re.compile(r"[。！？!?\.]+$")


def _clean_trailing_punct(text: str) -> str:
    return _TRAILING_PUNCT.sub("", text).strip()


def _check_empty(comment: GeneratedComment) -> Optional[str]:
    if not comment.text or len(comment.text.strip()) == 0:
        return "评论内容为空"
    return None


# ── Soft rules ───────────────────────────────────────────────────

_SOFT_THRESHOLD = 70  # rapidfuzz score
_JACCARD_THRESHOLD = 0.6


def _ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s: str) -> set[str]:
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    sa, sb = ngrams(a), ngrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _find_duplicates(
    comments: list[GeneratedComment],
) -> list[tuple[int, int]]:
    """Return index pairs of near-duplicate comments."""
    pairs: list[tuple[int, int]] = []
    texts = [c.text for c in comments]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = fuzz.token_sort_ratio(texts[i], texts[j])
            if score >= _SOFT_THRESHOLD:
                pairs.append((i, j))
                continue
            jac = _ngram_jaccard(texts[i], texts[j])
            if jac >= _JACCARD_THRESHOLD:
                pairs.append((i, j))
    return pairs


# ── Main validation entry ─────────────────────────────────────────


def validate_comments(
    comments: list[GeneratedComment],
    spec: BriefSpec,
    soft_flag_threshold: float = 0.10,
) -> tuple[list[GeneratedComment], ValidationResult]:
    """
    Apply hard and soft rules.
    Returns (validated_comments, ValidationResult).
    """
    min_len = spec.min_char_length
    forbidden = spec.forbidden_phrases

    # ── Hard rule pass ──────────────────────────────────────────
    validated: list[GeneratedComment] = []
    hard_failed = 0

    for c in comments:
        errors: list[str] = []

        # Auto-fix trailing punctuation
        c.text = _clean_trailing_punct(c.text)
        c.char_count = len(c.text)

        err = _check_empty(c)
        if err:
            errors.append(err)

        err = _check_length(c, min_len)
        if err:
            errors.append(err)

        err = _check_forbidden(c, forbidden)
        if err:
            errors.append(err)

        if errors:
            c.validation_status = "hard_fail"
            c.errors = errors
            hard_failed += 1
        else:
            c.validation_status = "pass"

        validated.append(c)

    # ── Soft rule: duplicate check ───────────────────────────────
    pass_comments = [c for c in validated if c.validation_status == "pass"]
    dup_pairs = _find_duplicates(pass_comments)

    # Mark duplicates (keep first, flag rest)
    flagged_indices: set[int] = set()
    for i, j in dup_pairs:
        flagged_indices.add(j)

    soft_flagged = 0
    for idx in flagged_indices:
        pass_comments[idx].validation_status = "soft_flag"
        pass_comments[idx].errors.append("与其他评论高度相似")
        soft_flagged += 1

    # Check soft flag rate
    total = len(validated)
    if total > 0 and soft_flagged / total > soft_flag_threshold:
        # Flag is acceptable; caller can decide to regenerate
        pass

    passed = sum(1 for c in validated if c.validation_status == "pass")

    result = ValidationResult(
        total=total,
        passed=passed,
        hard_failed=hard_failed,
        soft_flagged=soft_flagged,
        duplicate_pairs=dup_pairs,
    )
    return validated, result
