"""Report generator: assembles final Markdown report with citations."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.report.citations import extract_references, format_reference
from src.utils.config import get_config
from src.utils.logging import logger


def generate_report_file(
    report_markdown: str,
    research_results: list[dict],
    session_id: str,
    output_dir: str | None = None,
) -> str:
    """Write a complete research report to a Markdown file with references section.

    Returns the file path of the generated report.
    """
    cfg = get_config()
    out_dir = Path(output_dir) if output_dir else Path("data/reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build references section
    references = extract_references(research_results)
    ref_lines = [format_reference(i + 1, ref) for i, ref in enumerate(references)]
    ref_section = "\n".join(ref_lines)

    # Assemble full report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    full_report = (
        f"---\n"
        f"generated: {timestamp}\n"
        f"session: {session_id}\n"
        f"---\n\n"
        f"{report_markdown}\n\n"
        f"---\n\n"
        f"## References\n\n"
        f"{ref_section}\n"
    )

    filename = f"report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    filepath = out_dir / filename
    filepath.write_text(full_report, encoding="utf-8")

    logger.info(f"Report: saved to {filepath}")
    return str(filepath)
