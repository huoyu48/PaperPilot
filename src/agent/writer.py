"""Writer node: generates structured Markdown report with citations."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

SYSTEM_PROMPT = """\
You are a research report writer. Given a research question, a synthesis of findings, \
and the original source evidence, produce a professional Markdown research report.

## Report Structure

1. **Abstract** — 2-3 sentence executive summary
2. **Introduction** — Context, why this question matters, scope
3. **Methodology** — How research was conducted (tools used, search strategy)
4. **Key Findings** — Organized by theme, with specific data and citations
5. **Discussion** — Implications, limitations, open questions
6. **References** — Numbered list of all cited sources

## Citation Format

Use numbered inline citations [1], [2], etc. Every factual claim MUST be cited.

## Quality Rules

- Include specific numbers, dates, and technical details from sources
- Never hedge without evidence ("it is believed" → cite who believes it)
- If synthesis identifies gaps, explicitly state them in Discussion
- Keep the report between 1500-3000 words

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
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )

    logger.info("Writer: generating research report")

    results = state.get("research_results", [])
    sources_parts: list[str] = []
    for i, r in enumerate(results, 1):
        sources_parts.append(
            f"[{i}] {r['title']}\n    Tool: {r['tool']} | URL: {r['url']}\n"
            f"    {r['content'][:500]}"
        )
    sources_text = "\n\n".join(sources_parts)

    revision_context = ""
    if state.get("review_feedback") and state.get("needs_revision"):
        revision_context = (
            f"\n\n## Reviewer Feedback (revision #{state.get('revision_count', 0)})\n"
            f"{state['review_feedback']}\n"
            f"Please address ALL feedback points in the revised report."
        )

    prompt = (
        f"## Research Question\n{state['query']}\n\n"
        f"## Synthesis\n{state.get('synthesis', 'N/A')}\n\n"
        f"## Source Evidence\n{sources_text}"
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
        "report": response.content,
        "messages": [response],
    }
