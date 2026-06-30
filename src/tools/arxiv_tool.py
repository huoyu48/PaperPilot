"""Arxiv paper search tool using httpx for timeout control."""

from __future__ import annotations

import httpx
from langchain_core.tools import BaseTool
from xml.etree import ElementTree

from src.utils.logging import logger

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = "Search Arxiv for academic papers. Input: search query. Returns paper metadata and abstracts."

    max_results: int = 3
    timeout: int = 10

    def _run(self, query: str) -> list[dict]:
        logger.info(f"Arxiv: searching '{query[:60]}'")
        try:
            resp = httpx.get(
                ARXIV_API,
                params={
                    "search_query": f"all:{query}",
                    "max_results": self.max_results,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                },
                timeout=self.timeout,
                headers={"User-Agent": "PaperPilot/1.0"},
                follow_redirects=True,
            )
            resp.raise_for_status()

            root = ElementTree.fromstring(resp.text)
            results: list[dict] = []
            for entry in root.findall(f"{ATOM_NS}entry"):
                title = (entry.findtext(f"{ATOM_NS}title") or "").strip()
                summary = (entry.findtext(f"{ATOM_NS}summary") or "").strip()
                url = (entry.findtext(f"{ATOM_NS}id") or "").strip()
                published = (entry.findtext(f"{ATOM_NS}published") or "")[:10]

                authors: list[str] = []
                for author in entry.findall(f"{ATOM_NS}author"):
                    name = author.findtext(f"{ATOM_NS}name")
                    if name:
                        authors.append(name.strip())

                results.append({
                    "title": title,
                    "content": summary[:600],
                    "url": url,
                    "authors": ", ".join(authors[:3]),
                    "published": published,
                    "relevance": 0.8,
                })

            logger.info(f"Arxiv: found {len(results)} papers")
            return results
        except Exception as exc:
            logger.warning(f"Arxiv failed: {exc}")
            return [{"title": "ERROR", "content": f"Arxiv unavailable: {exc}", "url": "", "relevance": 0.0}]
