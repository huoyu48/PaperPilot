"""Reviewer node: quality check with revision feedback."""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

SYSTEM_PROMPT = """\
You are a strict research quality reviewer. Evaluate the report against the original \
question and source evidence.

## Checklist (score each 1-5)

1. **Completeness** — Does the report address all aspects of the question?
2. **Citation Coverage** — Is every factual claim backed by a cited source?
3. **Factual Accuracy** — Do claims match the source evidence?
4. **Specificity** — Does it include concrete numbers, names, and data?
5. **Structure** — Is the report well-organized and readable?

## Response Format (JSON only, no markdown fences)

{"approve": true/false, "scores": {"completeness":N, "citations":N, "accuracy":N, "specificity":N, "structure":N}, "feedback": "specific issues to fix, or empty if approved"}

Set "approve" to true ONLY if the average score >= 4 AND no individual score < 3.
"""


def reviewer_node(state: AgentState) -> dict:
    cfg = get_config()
    llm = create_llm(
        cfg.llm_provider, cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )

    logger.info(f"Reviewer: evaluating report (revision #{state.get('revision_count', 0)})")

    prompt = (
        f"## Original Question\n{state['query']}\n\n"
        f"## Report\n{state.get('report', '')}"
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", prompt),
    ])

    try:
        raw = response.content
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        review = json.loads(raw)
        approved = review.get("approve", True)
        feedback = review.get("feedback", "")
        scores = review.get("scores", {})
        logger.info(f"Reviewer: approved={approved}, scores={scores}")
    except (json.JSONDecodeError, IndexError):
        logger.warning("Reviewer: parse failed, auto-approving")
        approved = True
        feedback = ""

    revision_count = state.get("revision_count", 0)
    needs_revision = not approved and revision_count < 2

    return {
        "review_feedback": feedback,
        "needs_revision": needs_revision,
        "revision_count": revision_count + 1 if needs_revision else revision_count,
        "messages": [response],
    }


def should_revise(state: AgentState) -> str:
    """Conditional edge: route to 'revise' or 'done'."""
    if state.get("needs_revision"):
        logger.info("Reviewer: routing to revision")
        return "revise"
    logger.info("Reviewer: report approved")
    return "done"
