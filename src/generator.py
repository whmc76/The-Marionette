"""Generation engine: schedule tasks then batch-generate comments."""
from __future__ import annotations

import re
import uuid
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.models import BriefSpec, CommentCategory, CommentTask, GeneratedComment
from src.llm.client import LLMClient, GenerationAborted
from prompts.system_prompt import build_system_prompt
from prompts.comment_prompt import build_comment_prompt, PROMPT_VERSION


@dataclass
class GenerationProgress:
    total: int = 0
    success: int = 0
    failed: int = 0
    retried: int = 0
    elapsed: float = 0.0

    @property
    def done(self) -> int:
        return self.success + self.failed

    @property
    def eta_seconds(self) -> float:
        if self.done == 0 or self.elapsed == 0:
            return 0.0
        rate = self.done / self.elapsed
        remaining = self.total - self.done
        return remaining / rate if rate > 0 else 0.0


@dataclass
class GenerationResult:
    comments: list[GeneratedComment] = field(default_factory=list)
    progress: GenerationProgress = field(default_factory=GenerationProgress)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


# ── Phase A: Task scheduling ──────────────────────────────────────


def schedule_tasks(
    spec: BriefSpec,
    total_count: int,
    batch_size: int = 8,
) -> list[CommentTask]:
    """Distribute total_count across categories and personas."""
    if not spec.categories:
        return []

    # Split by direction ratio
    pos_categories = [c for c in spec.categories if c.direction == "正向"]
    other_categories = [c for c in spec.categories if c.direction != "正向"]

    pos_count = round(total_count * spec.positive_ratio)
    other_count = total_count - pos_count

    tasks: list[CommentTask] = []

    def _make_tasks_for_group(
        categories: list[CommentCategory], group_total: int
    ) -> list[CommentTask]:
        if not categories or group_total <= 0:
            return []
        per_cat = max(1, group_total // len(categories))
        remainder = group_total - per_cat * len(categories)
        group_tasks: list[CommentTask] = []

        for ci, cat in enumerate(categories):
            cat_total = per_cat + (1 if ci < remainder else 0)
            personas = cat.personas or ["真实用户"]

            per_persona = max(1, cat_total // len(personas))
            p_remainder = cat_total - per_persona * len(personas)

            for pi, persona in enumerate(personas):
                p_count = per_persona + (1 if pi < p_remainder else 0)
                if p_count <= 0:
                    continue
                # Split into batches
                batches = [batch_size] * (p_count // batch_size)
                leftover = p_count % batch_size
                if leftover:
                    batches.append(leftover)

                for bi, batch_count in enumerate(batches):
                    group_tasks.append(CommentTask(
                        task_id=str(uuid.uuid4()),
                        category=cat,
                        persona=persona,
                        target_count=batch_count,
                        batch_index=bi,
                    ))
        return group_tasks

    tasks.extend(_make_tasks_for_group(pos_categories, pos_count))
    tasks.extend(_make_tasks_for_group(other_categories, other_count))
    return tasks


# ── Phase B: Batch generation ─────────────────────────────────────


def run_generation(
    spec: BriefSpec,
    tasks: list[CommentTask],
    client: LLMClient,
    temperature: float = 0.9,
    on_progress: Optional[Callable[[GenerationProgress], None]] = None,
    on_comment: Optional[Callable[["GeneratedComment"], None]] = None,
) -> GenerationResult:
    """Execute all tasks and return collected results.

    on_comment is called immediately each time a single comment is parsed
    from the streaming response, enabling real-time UI updates.
    """
    system_prompt = build_system_prompt(spec)
    result = GenerationResult()
    result.progress.total = sum(t.target_count for t in tasks)
    already_generated: list[str] = []
    start_time = time.time()

    for task in tasks:
        if result.progress.failed > 0 and _is_aborted(result):
            break

        user_prompt = build_comment_prompt(task, already_generated)
        buf = _StreamBuffer()

        def _emit(text: str, _task: CommentTask = task) -> None:
            """Record one parsed comment and fire callbacks."""
            text = text.strip()
            if not text:
                return
            comment = GeneratedComment(
                task_id=_task.task_id,
                text=text,
                category_direction=_task.category.direction,
                theme=_task.category.theme,
                sub_theme=_task.category.sub_themes[0] if _task.category.sub_themes else "",
                persona=_task.persona,
                char_count=len(text),
            )
            result.comments.append(comment)
            already_generated.append(text)
            result.progress.success += 1
            result.progress.elapsed = time.time() - start_time
            if on_comment:
                on_comment(comment)
            if on_progress:
                on_progress(result.progress)

        def _on_chunk(chunk: str) -> None:
            for text in buf.feed(chunk):
                _emit(text)

        try:
            client.stream_chat(system_prompt, user_prompt, temperature, on_chunk=_on_chunk)
            for text in buf.flush():
                _emit(text)
        except GenerationAborted:
            result.progress.failed += task.target_count
            result.progress.elapsed = time.time() - start_time
            if on_progress:
                on_progress(result.progress)
            break
        except Exception:
            result.progress.failed += task.target_count
            result.progress.elapsed = time.time() - start_time
            if on_progress:
                on_progress(result.progress)

    result.progress.elapsed = time.time() - start_time
    return result


class _StreamBuffer:
    """Incrementally parses a numbered list from a streaming LLM response.

    Item N is considered complete once item N+1 appears in the buffer.
    Call flush() at end of stream to emit the final (possibly partial) item.
    """

    _COMPLETE = re.compile(
        r"^\s*\d+[\.、。]\s*(.+?)(?=\n\s*\d+[\.、。])",
        re.MULTILINE | re.DOTALL,
    )

    def __init__(self) -> None:
        self._buf = ""
        self._n_emitted = 0

    def feed(self, chunk: str) -> list[str]:
        """Append chunk; return any newly complete item texts."""
        self._buf += chunk
        matches = list(self._COMPLETE.finditer(self._buf))
        newly = matches[self._n_emitted:]
        self._n_emitted = len(matches)
        return [m.group(1).strip() for m in newly if m.group(1).strip()]

    def flush(self) -> list[str]:
        """End of stream — emit the last item that may not have a successor."""
        all_items = _parse_numbered_list(self._buf)
        new_items = all_items[self._n_emitted:]
        self._n_emitted = len(all_items)
        return [t for t in new_items if t.strip()]


def _parse_numbered_list(text: str) -> list[str]:
    """Extract lines from a numbered list (1. ... 2. ...) response."""
    lines = text.strip().splitlines()
    results: list[str] = []
    for line in lines:
        # Match "1. " prefix and capture rest
        m = re.match(r"^\s*\d+[\.、。]\s*(.+)", line)
        if m:
            results.append(m.group(1).strip())
        elif line.strip() and not re.match(r"^\s*\d+\s*$", line):
            # Non-numbered non-empty line—include if no numbered lines yet
            if not results:
                results.append(line.strip())
    return results


def _is_aborted(result: GenerationResult) -> bool:
    done = result.progress.done
    if done < 10:
        return False
    return result.progress.failed / done >= 0.5
