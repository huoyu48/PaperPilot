"""LangGraph workflow: the main research orchestration graph.

Flow:
    planner → researcher → writer → reviewer
                                     ├→ (revise) → writer
                                     └→ (done)   → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.agent.planner import planner_node
from src.agent.researcher import researcher_node
from src.agent.reviewer import reviewer_node, should_revise
from src.agent.state import AgentState
from src.agent.writer import writer_node
from src.utils.logging import logger


def build_graph() -> StateGraph:
    """Build and compile the research agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    # Linear pipeline
    graph.set_entry_point("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "reviewer")

    # Conditional: reviewer decides revise or done
    graph.add_conditional_edges("reviewer", should_revise, {
        "revise": "writer",
        "done": END,
    })

    return graph


_compiled_agent = None


def create_agent():
    """Create a compiled, invokable research agent (cached)."""
    global _compiled_agent
    if _compiled_agent is None:
        graph = build_graph()
        _compiled_agent = graph.compile()
        logger.info("Agent graph compiled: planner → researcher → writer → reviewer")
    return _compiled_agent
