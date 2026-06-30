from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """PaperPilot 全局配置，所有字段可通过 PAPERPILOT_ 前缀的环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAPERPILOT_",
        extra="ignore",
    )

    # --- LLM ---
    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 4096

    # --- Embedding ---
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_api_key: str = ""
    embedding_base_url: str = ""

    # --- Tool tokens ---
    github_token: str = ""
    search_max_results: int = 3

    # --- Storage ---
    chroma_path: str = "data/chroma"
    memory_path: str = "data/memory"

    # --- RAG ---
    chunk_size: int = 512
    chunk_overlap: int = 50

    # --- Memory ---
    max_memory_turns: int = 10

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    @property
    def data_dir(self) -> Path:
        p = Path("data")
        p.mkdir(exist_ok=True)
        return p


@lru_cache
def get_config() -> AppConfig:
    return AppConfig()
