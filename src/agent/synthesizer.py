"""Synthesizer node: cross-source analysis of research results."""

from __future__ import annotations

from langchain_core.messages import SystemMessage

from src.agent.state import AgentState
from src.llm.client import create_llm
from src.utils.config import get_config
from src.utils.logging import logger

SYSTEM_PROMPT = """\
You are a research synthesis expert. Given a research question and a collection of \
evidence from multiple sources (academic papers, code repositories, web articles), \
produce a structured synthesis.

Your synthesis MUST include:
1. **Key Themes**: 3-5 major themes identified across sources
2. **Converging Evidence**: where multiple sources agree
3. **Contradictions & Gaps**: where sources disagree or lack coverage
4. **Source Quality Assessment**: which sources are most authoritative and why
5. **Knowledge Map**: how findings connect to answer the original question

Use [Source N] notation to cite evidence. Be specific — include numbers, method names, \
and concrete details from the sources. Never say "some sources suggest" without citing.
"""


def synthesizer_node(state: AgentState) -> dict:
    cfg = get_config()
    llm = create_llm(
        cfg.llm_provider, cfg.llm_api_key, cfg.llm_base_url, cfg.llm_model,
        temperature=cfg.temperature, max_tokens=cfg.max_tokens,
    )

    results = state.get("research_results", [])
    logger.info(f"Synthesizer: analyzing {len(results)} results")

    evidence_parts: list[str] = []
    for i, r in enumerate(results, 1):
        evidence_parts.append(
            f"[Source {i}] ({r['tool']}) {r['title']}\n"
            f"URL: {r['url']}\n"
            f"{r['content'][:800]}"
        )
    evidence_text = "\n\n---\n\n".join(evidence_parts)

    prompt = f"Research Question: {state['query']}\n\nEvidence ({len(results)} sources):\n\n{evidence_text}"

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        ("human", prompt),
    ])

    logger.info("Synthesizer: cross-source analysis complete")
    return {
        "synthesis": response.content,
        "messages": [response],
    }
