from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Simple LRU-style cache for LLM clients
_llm_cache: dict = {}


def create_llm(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """Factory: 返回 LangChain ChatModel (cached to reuse HTTP connections).
    支持任何 OpenAI-compatible API (DeepSeek / Qwen DashScope / OpenAI)。
    """
    cache_key = (model, temperature, max_tokens)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _llm_cache[cache_key] = llm
    return llm


def create_embeddings(
    model_name: str = "BAAI/bge-small-zh-v1.5",
    api_key: str = "",
    base_url: str = "",
) -> Embeddings:
    """Factory: 返回 Embeddings 实例。
    默认使用本地 sentence-transformer；若提供 api_key+base_url 则走远程 API。
    """
    if api_key and base_url:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(api_key=api_key, base_url=base_url, model=model_name)

    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
