"""LangGraph State schema for the PaperPilot research workflow."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ──────────────────────────────────────────────
# Sub-structures
# ──────────────────────────────────────────────


class SubQuery(TypedDict):
    """A single sub-question decomposed from the user's research query."""
    id: str                         # "q1", "q2", ...
    question: str                   # 子问题
    tool: str                       # "arxiv" | "github" | "web_search" | "jina_reader"
    status: str                     # "pending" | "executed" | "failed"


class ResearchResult(TypedDict):
    """One piece of evidence gathered by a tool."""
    sub_query_id: str
    tool: str
    title: str
    content: str
    url: str
    relevance: float                # 0-1, LLM-scored


# ──────────────────────────────────────────────
# Main graph state
# ──────────────────────────────────────────────


class AgentState(TypedDict):
    """Shared state flowing through every node of the research graph."""

    # User input
    query: str
    session_id: str
    conversation_history: str  # formatted prior exchanges for follow-up context

    # Planner output
    sub_queries: list[SubQuery]

    # Researcher output
    research_results: list[ResearchResult]

    # Synthesizer output
    synthesis: str

    # Writer output
    report: str

    # Reviewer output
    review_feedback: str
    needs_revision: bool
    revision_count: int

    # LLM message history (accumulated via add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]

    # Errors
    errors: list[str]
