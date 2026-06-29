"""Document chunker: splits documents into overlapping chunks."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import get_config
from src.utils.logging import logger


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunks with configurable size and overlap."""
    cfg = get_config()
    size = chunk_size or cfg.chunk_size
    overlap = chunk_overlap or cfg.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"Chunker: {len(documents)} docs → {len(chunks)} chunks (size={size}, overlap={overlap})")
    return chunks
