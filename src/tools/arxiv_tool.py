"""Arxiv paper search tool."""

from __future__ import annotations

import arxiv
from langchain_core.tools import BaseTool

from src.utils.logging import logger


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = "Search Arxiv for academic papers. Input: search query. Returns paper metadata and abstracts."

    max_results: int = 3

    def _run(self, query: str) -> list[dict]:
        logger.info(f"Arxiv: searching '{query[:60]}'")
        try:
            client = arxiv.Client(page_size=self.max_results, delay_seconds=0.05)
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
            )

            results: list[dict] = []
            for paper in client.results(search):
                results.append({
                    "title": paper.title,
                    "content": paper.summary[:600],
                    "url": paper.entry_id,
                    "authors": ", ".join(a.name for a in paper.authors[:3]),
                    "published": str(paper.published.date()),
                    "relevance": 0.8,
                })

            logger.info(f"Arxiv: found {len(results)} papers")
            return results
        except Exception as exc:
            logger.warning(f"Arxiv failed: {exc}")
            return [{"title": "ERROR", "content": f"Arxiv unavailable: {exc}", "url": "", "relevance": 0.0}]
