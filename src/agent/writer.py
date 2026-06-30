"""Writer node: cross-source analysis + report generation in one LLM call."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

SYSTEM_PROMPT = """\
You are an expert research analyst and report writer. Given a research question and \
a collection of source evidence, first synthesize the findings across sources, then \
produce a professional Markdown research report.

## Synthesis Phase (internal — think before writing)

1. Identify 3-5 major themes across sources
2. Find converging evidence where multiple sources agree
3. Note contradictions and gaps
4. Assess source quality and relevance

## Report Structure

1. **摘要** — 2-3 sentence executive summary
2. **引言** — Context, why this question matters, scope
3. **主要发现** — Organized by theme, with specific data and citations
4. **讨论** — Implications, limitations, open questions
5. **参考文献** — Numbered list of all cited sources

## Citation Format

Use numbered inline citations [1], [2], etc. Every factual claim MUST be cited.

## Quality Rules

- Include specific numbers, dates, and technical details from sources
- Never hedge without evidence ("it is believed" → cite who believes it)
- If sources have gaps, explicitly state them in Discussion
- Keep the report between 1500-3000 words
- Write in the SAME language as the user's question

## Follow-up / Conversation Context

If CONVERSATION HISTORY is provided, this is a follow-up in an ongoing research session.
- If the user asks for a different language (e.g. "中文版", "用中文", "in Chinese"), \
  rewrite the report in that language while preserving all content and citations.
- If the user asks to expand on a section, focus the report on that section with more detail.
- If the user asks a new but related question, write a full report on the new question.
- Always write in the same language as the user's latest question.
"""


def writer_node(state: AgentState) -> dict:
    cfg = get_config()
    llm = create_llm(
        cfg.llm_provider, cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model,
        temperature=0.3, max_tokens=cfg.max_tokens,
    )

    logger.info("Writer: analyzing sources and generating report")

    results = state.get("research_results", [])
    sources_parts: list[str] = []
    for i, r in enumerate(results, 1):
        sources_parts.append(
            f"[Source {i}] ({r['tool']}) {r['title']}\n"
            f"URL: {r['url']}\n"
            f"{r['content'][:600]}"
        )
    sources_text = "\n\n---\n\n".join(sources_parts)

    revision_context = ""
    if state.get("review_feedback") and state.get("needs_revision"):
        revision_context = (
            f"\n\n## Reviewer Feedback (revision #{state.get('revision_count', 0)})\n"
            f"{state['review_feedback']}\n"
            f"Please address ALL feedback points in the revised report."
        )

    prompt = (
        f"## Research Question\n{state['query']}\n\n"
        f"## Source Evidence ({len(results)} sources)\n\n{sources_text}"
        f"{revision_context}"
    )

    # Inject conversation history for follow-up awareness
    history = state.get("conversation_history", "").strip()
    if history:
        prompt = f"## CONVERSATION HISTORY\n{history}\n\n---\n\n{prompt}"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", prompt),
    ])

    logger.info("Writer: report generated")
    return {
        "synthesis": "(integrated into report)",
        "report": response.content,
        "messages": [response],
    }
