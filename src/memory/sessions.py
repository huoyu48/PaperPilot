"""Session persistence: stores research sessions in SQLite for history browsing."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.config import get_config
from src.utils.logging import logger


class SessionStore:
    """SQLite-backed store for research session history."""

    def __init__(self, db_path: str | None = None):
        cfg = get_config()
        self._db_path = db_path or str(Path(cfg.memory_path) / "sessions.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    sub_queries TEXT DEFAULT '[]',
                    report TEXT DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS followups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    report TEXT DEFAULT '',
                    created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)

    def save_session(
        self,
        session_id: str,
        query: str,
        sub_queries: list[dict[str, Any]],
        report: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions (id, query, sub_queries, report)
                   VALUES (?, ?, ?, ?)""",
                (session_id, query, json.dumps(sub_queries, ensure_ascii=False), report),
            )
        logger.info(f"SessionStore: saved session {session_id}")

    def save_followup(self, session_id: str, query: str, report: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO followups (session_id, query, report) VALUES (?, ?, ?)",
                (session_id, query, report),
            )
        logger.info(f"SessionStore: saved followup for session {session_id}")

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, query, created_at FROM sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, query, sub_queries, report, created_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return None
            session = dict(row)
            session["sub_queries"] = json.loads(session["sub_queries"])

            # Load follow-ups
            followups = conn.execute(
                "SELECT query, report, created_at FROM followups WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            session["followups"] = [dict(f) for f in followups]
            return session

    def delete_session(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM followups WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        logger.info(f"SessionStore: deleted session {session_id}")
