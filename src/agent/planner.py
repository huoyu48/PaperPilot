"""Planner node: decomposes research query into structured sub-queries."""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState, SubQuery
from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

SYSTEM_PROMPT = """\
You are a senior research planner. Given a research question, decompose it into \
3-5 specific sub-queries that together answer the main question.

For each sub-query, choose exactly ONE tool:
- "arxiv"      — academic paper search (best for theory, methods, surveys)
- "github"     — code repository search (best for implementations, benchmarks)
- "web_search" — general web search (best for news, industry trends, product info)
- "jina_reader" — read a specific URL (use when you have a known URL to scrape)
- "local_docs" — search user-uploaded documents (use when the question asks about uploaded papers, files, or PDFs)

IMPORTANT: If the user's question mentions "上传的论文", "这篇论文", "上传的文档", "uploaded paper", or similar references to uploaded files, you MUST include at least one sub-query using "local_docs" tool.

## Follow-up Questions

If CONVERSATION HISTORY is provided below, this is a follow-up question in an ongoing session. \
Pay close attention to what was previously researched and reported.

Follow-up types and how to handle them:
- **Language/format change** (e.g. "我要中文版", "translate to Chinese", "用中文写"): \
  Create ONE sub-query using "web_search" with the original topic + "Chinese" keyword. \
  The writer will rewrite the report in the requested language using existing research.
- **Deepen a specific aspect** (e.g. "详细讲讲方法论部分"): Create sub-queries focused on that aspect.
- **New but related question**: Create normal sub-queries as if it were a new research task.

Rules:
- For arxiv queries about "latest" or "recent" topics, append "2024 OR 2025" to the query string
- If the user asks about latest trends/news, use web_search more than arxiv
- Write sub-query questions in the SAME language as the user's input
- Keep sub-queries specific and focused, not broad

Respond ONLY with a JSON array, no markdown fences:
[{"id":"q1","question":"...","tool":"arxiv"}, ...]
"""


def planner_node(state: AgentState) -> dict:
    cfg = get_config()
    llm = create_llm(
        cfg.llm_provider, cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )

    logger.info(f"Planner: decomposing query → {state['query'][:80]}...")

    # Build messages with optional conversation history
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    history = state.get("conversation_history", "").strip()
    if history:
        messages.append(("human", f"CONVERSATION HISTORY:\n{history}\n\n---\n\nNew question: {state['query']}"))
    else:
        messages.append(("human", state["query"]))

    response = llm.invoke(messages)

    try:
        raw = response.content
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        plan = json.loads(raw)
    except (json.JSONDecodeError, IndexError) as exc:
        logger.error(f"Planner JSON parse failed: {exc}")
        return {
            "sub_queries": [SubQuery(id="q1", question=state["query"], tool="web_search", status="pending")],
            "messages": [response],
            "errors": state.get("errors", []) + [f"Planner parse error: {exc}"],
        }

    sub_queries = [
        SubQuery(id=item["id"], question=item["question"], tool=item["tool"], status="pending")
        for item in plan
    ]
    logger.info(f"Planner: generated {len(sub_queries)} sub-queries")

    return {
        "sub_queries": sub_queries,
        "messages": [response],
    }
