"""Citation utilities: extract and format references from research results."""

from __future__ import annotations

from typing import Any


def extract_references(research_results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Deduplicate and extract reference info from research results."""
    seen_urls: set[str] = set()
    refs: list[dict[str, str]] = []

    for r in research_results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            refs.append({
                "title": r.get("title", "Untitled"),
                "url": url,
                "tool": r.get("tool", "unknown"),
                "content_preview": r.get("content", "")[:150],
            })

    return refs


def format_reference(index: int, ref: dict[str, str]) -> str:
    """Format a single reference as a Markdown list item."""
    title = ref.get("title", "Untitled")
    url = ref.get("url", "")
    tool = ref.get("tool", "")
    tool_badge = f" `[{tool}]`" if tool else ""

    if url:
        return f"{index}. [{title}]({url}){tool_badge}"
    return f"{index}. {title}{tool_badge}"


def inject_citation_markers(report: str, references: list[dict[str, str]]) -> str:
    """Verify that [N] markers in the report correspond to valid references.
    This is a validation function — it doesn't modify the report, just warns."""
    import re

    markers = set(int(m) for m in re.findall(r"\[(\d+)\]", report))
    valid = set(range(1, len(references) + 1))
    orphaned = markers - valid

    if orphaned:
        from src.utils.logging import logger
        logger.warning(f"Citations: orphaned markers {orphaned} (max ref = {len(references)})")

    return report
