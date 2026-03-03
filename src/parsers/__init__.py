"""Parser registry.

Priority:
  1. LLMParser  — if an LLMClient is provided (always preferred)
  2. Regex parsers — fallback when no LLM is configured
"""
from __future__ import annotations

from typing import Optional

from src.models import ParseReport
from src.parsers.classified import ClassifiedParser
from src.parsers.script_lib import ScriptLibParser
from src.parsers.fallback import FallbackParser

_REGEX_PARSERS = [ClassifiedParser(), ScriptLibParser(), FallbackParser()]


def parse_brief(
    path: str,
    llm_client=None,          # LLMClient | None
) -> tuple[ParseReport, list[ParseReport]]:
    """
    Parse a DOCX brief.
    - If llm_client is provided, use LLMParser (best result).
    - Always also run regex parsers and return all reports for reference.
    """
    reports: list[ParseReport] = []

    # ── LLM parse (primary) ──────────────────────────────────────
    if llm_client is not None:
        from src.parsers.llm_parser import LLMParser
        try:
            llm_report = LLMParser(llm_client).parse(path)
            reports.append(llm_report)
        except Exception as exc:
            reports.append(ParseReport(
                parser_name="llm",
                confidence=0.0,
                category_count=0, persona_count=0,
                forbidden_phrase_count=0, example_count=0,
                warnings=[f"LLM 解析器异常: {exc}"],
                spec=None,
            ))

    # ── Regex parsers (fallback / reference) ─────────────────────
    for parser in _REGEX_PARSERS:
        try:
            reports.append(parser.parse(path))
        except Exception as exc:
            reports.append(ParseReport(
                parser_name=parser.name,
                confidence=0.0,
                category_count=0, persona_count=0,
                forbidden_phrase_count=0, example_count=0,
                warnings=[f"解析器异常: {exc}"],
                spec=None,
            ))

    best = max(reports, key=lambda r: r.confidence)
    return best, reports
