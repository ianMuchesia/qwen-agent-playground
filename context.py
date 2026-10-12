"""Stage 1: a context assembler with a hard token budget.

Every request gets a fixed budget. Each section (retrieved docs, the question,
later: memory and tool schemas) gets its own cap, and when the total runs out
the lowest-priority sections are cut first. Nothing silently overflows.
"""
from __future__ import annotations

from dataclasses import dataclass


def count_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token for English.
    Exercise: swap this for Qwen's real tokenizer and see how far off it is."""
    return (len(text) + 3) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> tuple[str, bool]:
    if count_tokens(text) <= max_tokens:
        return text, False
    marker = " ...[truncated]"
    keep_chars = max(0, (max_tokens - count_tokens(marker)) * 4)
    cut = text[:keep_chars]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + marker, True


@dataclass
class Section:
    name: str
    text: str
    max_tokens: int
    priority: int = 0  # higher = protected first when the total budget is tight


def assemble(sections: list[Section], total_budget: int) -> tuple[str, list[dict]]:
    """Returns the assembled text (sections in original order) and a budget report."""
    remaining = total_budget
    kept: dict[int, str] = {}
    report: dict[int, dict] = {}
    for idx, s in sorted(enumerate(sections), key=lambda p: -p[1].priority):
        allowed = min(s.max_tokens, remaining)
        if allowed <= 0:
            report[idx] = {"section": s.name, "tokens": 0, "dropped": True}
            continue
        text, truncated = truncate_to_tokens(s.text, allowed)
        used = count_tokens(text)
        remaining -= used
        kept[idx] = text
        report[idx] = {"section": s.name, "tokens": used, "truncated": truncated}
    ordered = [kept[i] for i in sorted(kept)]
    return "\n\n".join(ordered), [report[i] for i in sorted(report)]
