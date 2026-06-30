"""Planner node: decomposes research query into structured sub-queries.

For simple new queries (no conversation history), uses a fast rule-based approach
that skips the LLM entirely — planning is instant. Only follow-up questions with
conversation context use the LLM planner.
"""

from __future__ import annotations

import json

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState, SubQuery
from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

# Shortened LLM prompt — only used for follow-up questions
LLM_PROMPT = """\
Decompose the research question into 3-4 sub-queries. Tools: arxiv, github, web_search, local_docs.
If "上传/这篇论文" mentioned, include a local_docs sub-query. Append "2024 2025" to arxiv queries about recent topics.
For follow-ups: language change → 1 web_search sub-query; deepening → focused sub-queries.
Reply JSON only: [{"id":"q1","question":"...","tool":"arxiv"}, ...]
"""


def _fast_plan(query: str) -> list[SubQuery]:
    """Rule-based instant planner — no LLM call needed."""
    q = query.strip()
    sub_queries: list[SubQuery] = []

    # Detect uploaded-doc references
    has_upload_ref = any(kw in q for kw in ["上传", "这篇", "文档", "uploaded", "this paper"])

    # Detect code/implementation intent
    has_code_intent = any(kw in q for kw in ["代码", "实现", "github", "code", "implement", "开源"])

    # Detect latest/trend intent
    has_latest_intent = any(kw in q for kw in ["最新", "近期", "发展", "动态", "趋势", "latest", "recent", "trend"])

    if has_upload_ref:
        sub_queries.append(SubQuery(id="q1", question=q, tool="local_docs", status="pending"))

    # ArXiv for academic/survey papers
    arxiv_q = f"{q} 综述 2024 2025" if has_latest_intent else f"{q} 研究论文"
    sub_queries.append(SubQuery(id=f"q{len(sub_queries)+1}", question=arxiv_q, tool="arxiv", status="pending"))

    # Web search for latest trends/news
    if has_latest_intent:
        web_q = f"{q} 最新进展 2024 2025"
        sub_queries.append(SubQuery(id=f"q{len(sub_queries)+1}", question=web_q, tool="web_search", status="pending"))
        sub_queries.append(SubQuery(id=f"q{len(sub_queries)+1}", question=f"{q} 应用案例和行业动态", tool="web_search", status="pending"))
    else:
        sub_queries.append(SubQuery(id=f"q{len(sub_queries)+1}", question=f"{q} 概述和应用", tool="web_search", status="pending"))

    # GitHub for code/implementation
    if has_code_intent:
        sub_queries.append(SubQuery(id=f"q{len(sub_queries)+1}", question=f"{q} 开源项目 实现", tool="github", status="pending"))

    # Multi-agent / collaboration sub-topic
    if "agent" in q.lower() or "智能体" in q:
        sub_queries.append(SubQuery(id=f"q{len(sub_queries)+1}", question=f"{q} 多智能体协作 2024 2025", tool="arxiv", status="pending"))

    # Cap at 4
    return sub_queries[:4]


def planner_node(state: AgentState) -> dict:
    query = state["query"]
    history = state.get("conversation_history", "").strip()

    # ── Fast path: no conversation history → rule-based, no LLM call ──
    if not history:
        sub_queries = _fast_plan(query)
        logger.info(f"Planner (fast): generated {len(sub_queries)} sub-queries instantly")
        return {"sub_queries": sub_queries, "messages": []}

    # ── LLM path: follow-up questions with conversation context ──
    cfg = get_config()
    llm = create_llm(
        cfg.llm_provider, cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model,
        temperature=0.2, max_tokens=256,
    )

    logger.info(f"Planner (LLM): follow-up query → {query[:80]}...")

    messages: list = [
        SystemMessage(content=LLM_PROMPT),
        ("human", f"CONVERSATION HISTORY:\n{history}\n\n---\n\nNew question: {query}"),
    ]

    response = llm.invoke(messages)

    try:
        raw = response.content
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        plan = json.loads(raw)
    except (json.JSONDecodeError, IndexError) as exc:
        logger.error(f"Planner JSON parse failed: {exc}, falling back to fast plan")
        return {
            "sub_queries": _fast_plan(query),
            "messages": [response],
            "errors": state.get("errors", []) + [f"Planner parse error: {exc}"],
        }

    sub_queries = [
        SubQuery(id=item["id"], question=item["question"], tool=item["tool"], status="pending")
        for item in plan
    ]
    logger.info(f"Planner (LLM): generated {len(sub_queries)} sub-queries")

    return {
        "sub_queries": sub_queries,
        "messages": [response],
    }
