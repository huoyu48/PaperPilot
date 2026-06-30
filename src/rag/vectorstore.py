"""Research vector store: ChromaDB-backed storage for research documents.

Uses a module-level singleton so the ChromaDB HTTP client is created once
and reused across requests. Re-creating the client per request leads to
"Cannot send a request, as the client has been closed" errors under
ThreadPoolExecutor (the client gets garbage-collected in the parent thread
while child threads are still using it).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag.embeddings import get_embeddings
from src.utils.config import get_config
from src.utils.logging import logger

# Module-level singleton — created once, reused by all ResearchVectorStore instances
_store: Chroma | None = None
_store_lock = __import__("threading").Lock()


def _get_store(collection_name: str, persist_dir: str) -> Chroma:
    """Return the singleton Chroma store, creating it on first call."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                embeddings = get_embeddings()
                Path(persist_dir).mkdir(parents=True, exist_ok=True)
                _store = Chroma(
                    collection_name=collection_name,
                    embedding_function=embeddings,
                    persist_directory=persist_dir,
                )
                logger.info(
                    f"VectorStore: initialized collection '{collection_name}' at {persist_dir}"
                )
    return _store


class ResearchVectorStore:
    """ChromaDB wrapper with session-aware storage. Shares a global Chroma instance."""

    def __init__(
        self,
        collection_name: str = "paperpilot_research",
        embeddings: Embeddings | None = None,
        persist_dir: str | None = None,
    ):
        cfg = get_config()
        self._persist_dir = persist_dir or cfg.chroma_path
        # embeddings param accepted for API compatibility but the singleton
        # uses whatever was configured on first init.
        self._store = _get_store(collection_name, self._persist_dir)

    def add_documents(
        self,
        documents: list[Document],
        session_id: str | None = None,
    ) -> list[str]:
        """Add documents to the vector store. Tags with session_id if provided."""
        if session_id:
            for doc in documents:
                doc.metadata["session_id"] = session_id

        ids = self._store.add_documents(documents)
        logger.info(f"VectorStore: added {len(documents)} docs (session={session_id})")
        return ids

    def search(
        self,
        query: str,
        top_k: int = 5,
        session_id: str | None = None,
    ) -> list[Document]:
        """Similarity search, optionally filtered by session."""
        where: dict[str, Any] | None = {"session_id": session_id} if session_id else None
        results = self._store.similarity_search(query, k=top_k, filter=where)
        logger.debug(f"VectorStore: search '{query[:40]}' → {len(results)} results")
        return results

    def count(self) -> int:
        """Return total document count in the collection."""
        return self._store._collection.count()
