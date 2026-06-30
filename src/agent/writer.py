"""Writer node: cross-source analysis + report generation in one LLM call.

Supports streaming output: when a stream callback is registered (by the WebSocket
handler), the writer uses llm.stream() and sends partial chunks to the frontend
in real-time, so the user sees the report being written instead of waiting.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

# Module-level callback — set by WebSocket handler to enable streaming
_stream_callback = None


def set_stream_callback(cb):
    """Register a callback(chunk_text: str) to receive streamed report chunks."""
    global _stream_callback
    _stream_callback = cb


SYSTEM_PROMPT = """\
You are an expert research analyst and report writer. Given a research question and \
a collection of source evidence, first synthesize the findings across sources, then \
produce a professional Markdown research report.

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
- Keep the report between 1500-2500 words
- Write in the SAME language as the user's question

## Follow-up / Conversation Context

If CONVERSATION HISTORY is provided, this is a follow-up in an ongoing research session.
- If the user asks for a different language (e.g. "中文版", "用中文"), rewrite in that language.
- If the user asks to expand on a section, focus the report on that section.
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

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", prompt),
    ]

    # Use streaming if a callback is registered, otherwise blocking invoke
    if _stream_callback:
        full_text = ""
        for chunk in llm.stream(messages):
            full_text += chunk.content
            if chunk.content:
                _stream_callback(chunk.content)
        logger.info("Writer: report generated (streamed)")
        return {
            "synthesis": "(integrated into report)",
            "report": full_text,
            "messages": [],
        }
    else:
        response = llm.invoke(messages)
        logger.info("Writer: report generated")
        return {
            "synthesis": "(integrated into report)",
            "report": response.content,
            "messages": [response],
        }
