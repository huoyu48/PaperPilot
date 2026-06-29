"""Summary memory: LLM-powered conversation and research summarization."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage

from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

SUMMARY_PROMPT = """\
You are a research assistant that creates concise summaries. \
Given a conversation or research session, produce a summary that captures:
1. The main question/topic explored
2. Key findings and conclusions
3. Open questions or next steps

Keep it under 200 words. Be specific with facts, numbers, and source names.
"""


class SummaryMemory:
    """Generates LLM summaries of research sessions for long-term storage."""

    def __init__(self):
        self._cfg = get_config()

    def summarize_messages(self, messages: list[BaseMessage]) -> str:
        """Summarize a conversation thread."""
        llm = create_llm(
            self._cfg.llm_provider,
            self._cfg.llm_api_key,
            self._cfg.llm_base_url,
            self._cfg.llm_model,
            temperature=0.3,  # Lower temp for summary
        )

        # Take last 10 messages for context
        recent = messages[-10:]
        conv_text = "\n".join(
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content[:300]}"
            for m in recent
        )

        response = llm.invoke([
            SystemMessage(content=SUMMARY_PROMPT),
            ("human", f"Summarize this research conversation:\n\n{conv_text}"),
        ])

        logger.info("SummaryMemory: conversation summary generated")
        return response.content

    def summarize_research(self, query: str, report: str) -> str:
        """Summarize a completed research report."""
        llm = create_llm(
            self._cfg.llm_provider,
            self._cfg.llm_api_key,
            self._cfg.llm_base_url,
            self._cfg.llm_model,
            temperature=0.3,
        )

        response = llm.invoke([
            SystemMessage(content=SUMMARY_PROMPT),
            ("human", f"Research question: {query}\n\nReport:\n{report[:2000]}"),
        ])

        logger.info("SummaryMemory: research summary generated")
        return response.content
