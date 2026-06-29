"""Research retriever: high-level interface for RAG retrieval."""

from __future__ import annotations

from langchain_core.documents import Document

from src.rag.vectorstore import ResearchVectorStore
from src.utils.logging import logger


class ResearchRetriever:
    """Wraps ResearchVectorStore with a simple retrieval interface."""

    def __init__(self, vectorstore: ResearchVectorStore | None = None):
        self._store = vectorstore or ResearchVectorStore()

    def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[Document]:
        """Retrieve relevant documents for a query."""
        logger.info(f"Retriever: query='{query[:50]}', session={session_id}, top_k={top_k}")
        return self._store.search(query, top_k=top_k, session_id=session_id)

    def add_and_retrieve(
        self,
        documents: list[Document],
        query: str,
        session_id: str | None = None,
        top_k: int = 5,
    ) -> list[Document]:
        """Add new documents then retrieve — useful for incremental research sessions."""
        self._store.add_documents(documents, session_id=session_id)
        return self.retrieve(query, session_id=session_id, top_k=top_k)
