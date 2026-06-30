"""Researcher node: executes sub-queries via tool dispatch."""

from __future__ import annotations

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from src.agent.state import AgentState, ResearchResult, SubQuery
from src.rag.retriever import ResearchRetriever
from src.tools.arxiv_tool import ArxivSearchTool
from src.tools.github_tool import GitHubSearchTool
from src.tools.web_search_tool import WebSearchTool
from src.tools.jina_reader import JinaReaderTool
from src.utils.logging import logger

_TOOLS = {
    "arxiv": ArxivSearchTool,
    "github": GitHubSearchTool,
    "web_search": WebSearchTool,
    "jina_reader": JinaReaderTool,
}


def _search_local_docs(query: str, top_k: int = 5) -> list[dict]:
    """Query the local RAG store for uploaded documents.

    Wrapped in a hard 10s timeout — RAG should never block the pipeline.
    Uses a daemon thread so a stuck retrieval doesn't block process exit.
    """
    import concurrent.futures
    import threading

    result_holder: list[list[dict]] = []

    def _do_retrieve() -> None:
        try:
            retriever = ResearchRetriever()
            docs = retriever.retrieve(query, top_k=top_k)
            if not docs:
                logger.info("RAG: no local documents found")
                result_holder.append([])
                return
            results = []
            for i, doc in enumerate(docs):
                results.append({
                    "title": doc.metadata.get("source", f"上传文档 #{i+1}"),
                    "content": doc.page_content[:1500],
                    "url": "",
                    "relevance": 0.9,
                })
            logger.info(f"RAG: retrieved {len(results)} local chunks")
            result_holder.append(results)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")
            result_holder.append([])

    try:
        t = threading.Thread(target=_do_retrieve, daemon=True)
        t.start()
        t.join(timeout=10)
        if t.is_alive():
            logger.warning("RAG retrieval timed out (10s), skipping")
            return []
        return result_holder[0] if result_holder else []
    except Exception as exc:
        logger.warning(f"RAG retrieval failed: {exc}")
        return []


def _execute_single(sq: SubQuery) -> list[ResearchResult]:
    """Run one sub-query and return a list of results."""
    # Handle local_docs tool specially
    if sq["tool"] == "local_docs":
        raw = _search_local_docs(sq["question"])
        if not raw:
            return [ResearchResult(
                sub_query_id=sq["id"], tool="local_docs",
                title="NO_LOCAL_DOCS",
                content="未找到上传文档，请确认是否已上传文件。",
                url="", relevance=0.0,
            )]
        results: list[ResearchResult] = []
        for item in raw:
            results.append(ResearchResult(
                sub_query_id=sq["id"], tool="local_docs",
                title=item["title"], content=item["content"],
                url=item["url"], relevance=item["relevance"],
            ))
        return results

    tool_cls = _TOOLS.get(sq["tool"])
    if not tool_cls:
        logger.warning(f"Unknown tool: {sq['tool']}, falling back to web_search")
        tool_cls = _TOOLS["web_search"]

    tool = tool_cls()
    try:
        raw = tool.run(sq["question"])
    except Exception as exc:
        logger.error(f"Tool {sq['tool']} failed for '{sq['question'][:60]}': {exc}")
        return [ResearchResult(
            sub_query_id=sq["id"],
            tool=sq["tool"],
            title="ERROR",
            content=str(exc),
            url="",
            relevance=0.0,
        )]

    results: list[ResearchResult] = []
    for i, item in enumerate(raw[:5]):  # cap at 5 per sub-query
        results.append(ResearchResult(
            sub_query_id=sq["id"],
            tool=sq["tool"],
            title=item.get("title", ""),
            content=item.get("content", item.get("snippet", "")),
            url=item.get("url", ""),
            relevance=item.get("relevance", 0.5),
        ))
    return results


def researcher_node(state: AgentState) -> dict:
    """Execute all pending sub-queries and collect results."""
    pending = [sq for sq in state.get("sub_queries", []) if sq["status"] == "pending"]
    logger.info(f"Researcher: executing {len(pending)} sub-queries")

    all_results = list(state.get("research_results", []))
    updated_queries = list(state.get("sub_queries", []))
    errors = list(state.get("errors", []))

    # Auto-inject RAG retrieval for the main query (always, even if no local_docs sub-query)
    has_local_docs_sq = any(sq["tool"] == "local_docs" for sq in pending)
    if not has_local_docs_sq:
        local_results = _search_local_docs(state["query"], top_k=3)
        if local_results:
            logger.info(f"Researcher: auto-injected {len(local_results)} RAG results")
            for item in local_results:
                all_results.append(ResearchResult(
                    sub_query_id="rag_auto",
                    tool="local_docs",
                    title=item["title"],
                    content=item["content"],
                    url=item["url"],
                    relevance=item["relevance"],
                ))

    # Execute sub-queries in parallel with 20s hard timeout (was 30s).
    # Tools have their own per-call timeouts (6s for web, 10s for arxiv/github),
    # so 20s gives enough headroom while not blocking the pipeline too long.
    if pending:
        pool = ThreadPoolExecutor(max_workers=min(len(pending), 5))
        future_to_sq = {pool.submit(_execute_single, sq): sq for sq in pending}
        done, not_done = concurrent.futures.wait(
            future_to_sq.keys(), timeout=20, return_when=concurrent.futures.ALL_COMPLETED,
        )
        for future in done:
            sq = future_to_sq[future]
            try:
                results = future.result()
            except Exception as exc:
                logger.error(f"Tool {sq['tool']} crashed: {exc}")
                results = [ResearchResult(
                    sub_query_id=sq["id"], tool=sq["tool"],
                    title="ERROR", content=str(exc), url="", relevance=0.0,
                )]
            all_results.extend(results)
            sq["status"] = "failed" if all(r["title"] == "ERROR" for r in results) else "executed"
        # Handle timed-out futures
        for future in not_done:
            sq = future_to_sq[future]
            logger.warning(f"Tool {sq['tool']} timed out (20s) for '{sq['question'][:60]}'")
            all_results.append(ResearchResult(
                sub_query_id=sq["id"], tool=sq["tool"],
                title="TIMEOUT", content=f"搜索超时(20秒)", url="", relevance=0.0,
            ))
            sq["status"] = "failed"
        pool.shutdown(wait=False, cancel_futures=True)

    # Update status in the list (dicts are mutable but we re-assign for clarity)
    for i, sq in enumerate(updated_queries):
        for done in pending:
            if sq["id"] == done["id"]:
                updated_queries[i] = done

    logger.info(f"Researcher: collected {len(all_results)} results total")

    return {
        "research_results": all_results,
        "sub_queries": updated_queries,
        "errors": errors,
    }
