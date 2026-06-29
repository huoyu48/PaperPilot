"""Embedding factory: wraps src.llm.client.create_embeddings for convenience."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from src.llm.client import create_embeddings as _create
from src.utils.config import get_config


def get_embeddings() -> Embeddings:
    """Return the configured Embeddings instance."""
    cfg = get_config()
    return _create(cfg.embedding_model, cfg.embedding_api_key, cfg.embedding_base_url)
