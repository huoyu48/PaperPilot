"""Web search tool using DuckDuckGo (no API key required)."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from src.utils.config import get_config
from src.utils.logging import logger


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the web using DuckDuckGo. Input: search query. Returns web page snippets."

    def _run(self, query: str) -> list[dict]:
        cfg = get_config()
        logger.info(f"WebSearch: searching '{query[:60]}'")

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=cfg.search_max_results))

            results: list[dict] = []
            for item in raw:
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("body", ""),
                    "url": item.get("href", ""),
                    "relevance": 0.6,
                })
            logger.info(f"WebSearch: found {len(results)} results")
            return results
        except ImportError:
            logger.warning("duckduckgo-search not installed, trying httpx fallback")
            return self._fallback_httpx(query, cfg.search_max_results)
        except Exception as exc:
            logger.warning(f"WebSearch failed: {exc}")
            return [{"title": "ERROR", "content": f"Search unavailable: {exc}", "url": "", "relevance": 0.0}]

    def _fallback_httpx(self, query: str, max_results: int) -> list[dict]:
        """Fallback: use DuckDuckGo HTML endpoint via httpx."""
        import httpx

        try:
            resp = httpx.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            # Very basic extraction — won't be perfect but better than nothing
            from html.parser import HTMLParser

            class SnippetParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.results: list[dict] = []
                    self._in_result = False
                    self._current: dict = {}
                    self._capture = ""

                def handle_starttag(self, tag, attrs):
                    d = dict(attrs)
                    if tag == "a" and "result__a" in d.get("class", ""):
                        self._in_result = True
                        self._current = {"url": d.get("href", ""), "title": "", "content": ""}
                        self._capture = ""

                def handle_data(self, data):
                    if self._in_result:
                        self._capture += data

                def handle_endtag(self, tag):
                    if self._in_result and tag == "a":
                        self._current["title"] = self._capture.strip()
                        self._in_result = False
                        if self._current["title"]:
                            self.results.append(self._current)

            parser = SnippetParser()
            parser.feed(resp.text)

            results = []
            for r in parser.results[:max_results]:
                results.append({
                    "title": r["title"],
                    "content": r.get("content", ""),
                    "url": r["url"],
                    "relevance": 0.5,
                })
            logger.info(f"WebSearch fallback: found {len(results)} results")
            return results
        except Exception as exc:
            logger.warning(f"WebSearch fallback failed: {exc}")
            return [{"title": "ERROR", "content": str(exc), "url": "", "relevance": 0.0}]
