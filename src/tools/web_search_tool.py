"""Web search tool using Sogou (no API key required).

DuckDuckGo, Jina, and Google are unreachable from many networks; Baidu blocks
httpx with a CAPTCHA page; Bing mishandles Chinese queries (splits them into
single characters). Sogou is fast, reliable, understands both Chinese and
English queries correctly, and works with httpx out of the box.
"""

from __future__ import annotations

import re

import httpx
from langchain_core.tools import BaseTool

from src.utils.config import get_config
from src.utils.logging import logger

_SOGOU_URL = "https://www.sogou.com/web"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_TAG_RE = re.compile(r"<[^>]+>")
_H3_RE = re.compile(r'<h3[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL)


def _strip_tags(s: str) -> str:
    s = _TAG_RE.sub("", s)
    s = s.replace("&amp;", "&").replace("&ensp;", " ").replace("&#0183;", "·")
    s = s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    return re.sub(r"\s+", " ", s).strip()


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web via Sogou. Works for both Chinese and English queries. "
        "Returns web page titles and snippets."
    )

    timeout: int = 8

    def _run(self, query: str) -> list[dict]:
        cfg = get_config()
        max_results = cfg.search_max_results
        logger.info(f"WebSearch: '{query[:60]}'")

        try:
            resp = httpx.get(
                _SOGOU_URL,
                params={"query": query, "num": str(max_results + 5)},
                headers={"User-Agent": _USER_AGENT, "Accept-Language": "zh-CN,zh;q=0.9"},
                timeout=self.timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
            html = resp.text
        except Exception as exc:
            logger.warning(f"WebSearch failed: {exc}")
            return [{"title": "ERROR", "content": f"Search unavailable: {exc}", "url": "", "relevance": 0.0}]

        results: list[dict] = []
        for m in _H3_RE.finditer(html):
            url = m.group(1)
            if url.startswith("/"):
                url = "https://www.sogou.com" + url
            title = _strip_tags(m.group(2))
            if not title or len(title) < 4:
                continue
            # Extract snippet: text after h3, skip scripts/styles
            after = html[m.end():m.end() + 2000]
            after = re.sub(r"<script.*?</script>", "", after, flags=re.DOTALL)
            after = re.sub(r"<style.*?</style>", "", after, flags=re.DOTALL)
            text = _strip_tags(after)
            snippet = text[:300] if len(text) > 20 else title
            results.append({
                "title": title,
                "content": snippet,
                "url": url,
                "relevance": 0.6,
            })
            if len(results) >= max_results:
                break

        logger.info(f"WebSearch: found {len(results)} results")

        if not results:
            return [{"title": "ERROR", "content": "No results found", "url": "", "relevance": 0.0}]
        return results
