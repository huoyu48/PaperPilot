"""Long-term memory: persistent research profile stored in SQLite."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.logging import logger


class ResearchProfile:
    """Stores user's research interests and history across sessions."""

    def __init__(self, db_path: str | None = None):
        cfg = get_config()
        self._db_path = db_path or str(Path(cfg.memory_path) / "research_profile.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    topic TEXT NOT NULL,
                    session_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    summary TEXT,
                    report_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def add_topic(self, topic: str, session_id: str = "", metadata: dict | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO research_topics (topic, session_id, metadata) VALUES (?, ?, ?)",
                (topic, session_id, json.dumps(metadata or {})),
            )
        logger.debug(f"LongTermMemory: added topic '{topic}'")

    def get_topics(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT topic, session_id, created_at, metadata FROM research_topics ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"topic": r[0], "session_id": r[1], "created_at": r[2], "metadata": json.loads(r[3])}
            for r in rows
        ]

    def save_summary(
        self, session_id: str, query: str, summary: str, report_path: str = ""
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO research_summaries (session_id, query, summary, report_path) VALUES (?, ?, ?, ?)",
                (session_id, query, summary, report_path),
            )

    def get_summaries(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, query, summary, report_path, created_at FROM research_summaries ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"session_id": r[0], "query": r[1], "summary": r[2], "report_path": r[3], "created_at": r[4]}
            for r in rows
        ]

    def get_context_for_query(self, query: str) -> str:
        """Return relevant past research context for a new query."""
        topics = self.get_topics(limit=10)
        summaries = self.get_summaries(limit=5)

        parts: list[str] = []
        if topics:
            topic_list = ", ".join(t["topic"] for t in topics[:5])
            parts.append(f"Past research topics: {topic_list}")
        if summaries:
            for s in summaries[:3]:
                parts.append(f"Past research: \"{s['query'][:60]}\" → {s['summary'][:100]}")

        return "\n".join(parts) if parts else "No prior research context."
