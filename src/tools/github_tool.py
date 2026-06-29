"""GitHub repository search tool."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from src.utils.config import get_config
from src.utils.logging import logger


class GitHubSearchTool(BaseTool):
    name: str = "github_search"
    description: str = "Search GitHub for repositories. Input: search query. Returns repo metadata and descriptions."

    max_results: int = 3

    def _run(self, query: str) -> list[dict]:
        cfg = get_config()
        logger.info(f"GitHub: searching '{query[:60]}'")

        try:
            from github import Github
            gh = Github(cfg.github_token) if cfg.github_token else Github()
        except ImportError:
            logger.warning("PyGithub not installed")
            return [{"title": "ERROR", "content": "PyGithub not installed", "url": "", "relevance": 0.0}]

        try:
            repos = gh.search_repositories(query=query, sort="stars", order="desc")
            results: list[dict] = []
            for i, repo in enumerate(repos):
                if i >= self.max_results:
                    break
                results.append({
                    "title": f"{repo.full_name} ({repo.stargazers_count} stars)",
                    "content": repo.description or "No description",
                    "url": repo.html_url,
                    "language": repo.language or "unknown",
                    "stars": repo.stargazers_count,
                    "relevance": min(0.9, repo.stargazers_count / 10000 + 0.3),
                })
            logger.info(f"GitHub: found {len(results)} repos")
            return results
        except Exception as exc:
            logger.warning(f"GitHub search failed: {exc}")
            return [{"title": "ERROR", "content": f"GitHub unavailable: {exc}", "url": "", "relevance": 0.0}]
