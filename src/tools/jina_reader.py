"""Jina Reader tool: fetches clean text from a URL via Jina Reader API."""

from __future__ import annotations

import httpx
from langchain_core.tools import BaseTool

from src.utils.logging import logger

JINA_READER_BASE = "https://r.jina.ai/"


class JinaReaderTool(BaseTool):
    name: str = "jina_reader"
    description: str = "Read and extract clean text from a web URL using Jina Reader. Input: a full URL. Returns the page text content."

    timeout: int = 30

    def _run(self, url: str) -> list[dict]:
        url = url.strip()
        if not url.startswith("http"):
            url = "https://" + url

        logger.info(f"JinaReader: fetching {url[:80]}")

        try:
            resp = httpx.get(
                f"{JINA_READER_BASE}{url}",
                timeout=self.timeout,
                headers={"Accept": "text/plain"},
            )
            resp.raise_for_status()
            text = resp.text[:4000]  # Cap content length
        except Exception as exc:
            logger.error(f"JinaReader fetch failed: {exc}")
            return [{"title": "ERROR", "content": str(exc), "url": url, "relevance": 0.0}]

        return [{
            "title": f"Web page: {url}",
            "content": text,
            "url": url,
            "relevance": 0.7,
        }]
