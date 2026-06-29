"""Research vector store: ChromaDB-backed storage for research documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.rag.embeddings import get_embeddings
from src.utils.config import get_config
from src.utils.logging import logger


class ResearchVectorStore:
    """ChromaDB wrapper with session-aware storage."""

    def __init__(
        self,
        collection_name: str = "paperpilot_research",
        embeddings: Embeddings | None = None,
        persist_dir: str | None = None,
    ):
        cfg = get_config()
        self._embeddings = embeddings or get_embeddings()
        self._persist_dir = persist_dir or cfg.chroma_path

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )
        logger.info(f"VectorStore: initialized collection '{collection_name}' at {self._persist_dir}")

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
