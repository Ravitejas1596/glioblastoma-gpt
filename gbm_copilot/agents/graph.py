"""
LangGraph StateGraph — GBM Copilot HITL Multi-Agent Graph
Wires all agents into a conditional parallel fan-out graph.

Topology:
START → triage_agent
triage → [research | trial | drug | emotional] (based on query_type)
                        ↓ (all routes)
               synthesizer_agent
                        ↓
                  safety_layer
                        ↓
                       END
"""
from __future__ import annotations

from typing import Literal

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from gbm_copilot.agents.state import AgentState
from gbm_copilot.agents.triage_agent import triage_agent
from gbm_copilot.agents.research_agent import research_agent
from gbm_copilot.agents.clinical_trial_agent import clinical_trial_agent
from gbm_copilot.agents.drug_agent import drug_agent
from gbm_copilot.agents.emotional_support_agent import emotional_support_agent, is_emotional_query
from gbm_copilot.agents.synthesizer_agent import synthesizer_agent
from gbm_copilot.agents.safety_layer import safety_layer


def route_after_triage(state: AgentState) -> list[str]:
    """
    Conditional router: determines which agents to activate after triage.
    Can activate multiple agents in parallel.
    """
    query_type = state.get("query_type", "research")
    query = state.get("query", "")
    routes: list[str] = []

    # Primary routing
    if query_type == "emotional":
        routes.append("emotional_support")
        # Also get some research info for emotional queries that mention treatments
        if any(w in query.lower() for w in ["treatment", "trial", "drug", "diagnosis"]):
            routes.append("research")
    elif query_type == "trial":
        routes.append("clinical_trial")
        routes.append("research")  # Context from research helps explain trials
    elif query_type == "drug":
        routes.append("drug")
        routes.append("research")  # Research for drug context
    elif query_type == "treatment":
        routes.append("research")
        # Check for trial mentions
        if any(w in query.lower() for w in ["trial", "experimental", "study", "research"]):
            routes.append("clinical_trial")
    else:  # "research" or default
        routes.append("research")

    # Always check for emotional undertones (add emotional support if detected)
    if query_type != "emotional" and is_emotional_query(query):
        if "emotional_support" not in routes:
            routes.append("emotional_support")

    return routes


def build_graph(checkpointing: bool = True) -> StateGraph:
    """
    Build and compile the GBM Copilot LangGraph.
    
    Args:
        checkpointing: Enable memory-based checkpointing for multi-turn conversations
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("triage", triage_agent)
    graph.add_node("research", research_agent)
    graph.add_node("clinical_trial", clinical_trial_agent)
    graph.add_node("drug", drug_agent)
    graph.add_node("emotional_support", emotional_support_agent)
    graph.add_node("synthesizer", synthesizer_agent)
    graph.add_node("safety", safety_layer)

    # Entry point
    graph.add_edge(START, "triage")

    # Conditional parallel fan-out from triage
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "research": "research",
            "clinical_trial": "clinical_trial",
            "drug": "drug",
            "emotional_support": "emotional_support",
        },
    )

    # All specialized agents converge to synthesizer
    graph.add_edge("research", "synthesizer")
    graph.add_edge("clinical_trial", "synthesizer")
    graph.add_edge("drug", "synthesizer")
    graph.add_edge("emotional_support", "synthesizer")

    # Synthesizer → safety layer → end
    graph.add_edge("synthesizer", "safety")
    graph.add_edge("safety", END)

    # Compile
    if checkpointing:
        memory = MemorySaver()
        return graph.compile(checkpointer=memory)
    else:
        return graph.compile()


# Module-level compiled graph (lazy init)
_graph = None


def get_graph():
    """Get or build the compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointing=True)
    return _graph


async def run_query(
    query: str,
    literacy_mode: str = "patient",
    session_id: str = "default",
    conversation_history: list[dict] | None = None,
) -> AgentState:
    """
    Main entry point: run the full agent graph for a query.
    Returns the final AgentState with all results.
    """
    import uuid
    graph = get_graph()

    initial_state: AgentState = {
        "query": query,
        "expanded_query": "",
        "literacy_mode": literacy_mode,
        "conversation_history": conversation_history or [],
        "query_type": "",
        "triage_confidence": 0.0,
        "research_results": [],
        "research_answer": "",
        "trial_results": [],
        "trial_answer": "",
        "drug_results": [],
        "drug_answer": "",
        "emotional_answer": "",
        "emotional_resources": [],
        "final_answer": "",
        "citations": [],
        "confidence_score": 0.0,
        "safety_flags": [],
        "safety_disclaimer": "",
        "is_blocked": False,
        "session_id": session_id,
        "trace_url": "",
        "error": "",
    }

    config = {"configurable": {"thread_id": session_id}}

    final_state = await graph.ainvoke(initial_state, config=config)
    return final_state
