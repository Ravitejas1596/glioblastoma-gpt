"""
LangGraph Agent State
TypedDict that flows through all nodes in the GBM Copilot agent graph.
"""
from __future__ import annotations

from typing import Annotated, TypedDict
import operator


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                      # Original user query
    expanded_query: str             # After medical query expansion
    literacy_mode: str              # "patient" | "caregiver" | "clinician"
    conversation_history: list[dict]  # [{role, content}, ...]

    # ── Triage ────────────────────────────────────────────────────────────────
    query_type: str                 # treatment|trial|drug|emotional|research
    triage_confidence: float

    # ── Research Agent ────────────────────────────────────────────────────────
    research_results: list[dict]    # Retrieved chunks from hybrid retriever
    research_answer: str

    # ── Clinical Trial Agent ──────────────────────────────────────────────────
    trial_results: list[dict]       # From clinicaltrials.gov
    trial_answer: str

    # ── Drug Agent ────────────────────────────────────────────────────────────
    drug_results: list[dict]        # From FDA labels + ontology
    drug_answer: str

    # ── Emotional Support Agent ───────────────────────────────────────────────
    emotional_answer: str
    emotional_resources: list[dict]

    # ── Synthesizer ───────────────────────────────────────────────────────────
    final_answer: str
    citations: list[dict]           # [{title, url, source, year}, ...]
    confidence_score: float

    # ── Safety Layer ──────────────────────────────────────────────────────────
    safety_flags: list[str]         # List of triggered safety rules
    safety_disclaimer: str          # Appended disclaimer if triggered
    is_blocked: bool                # True if answer is blocked entirely

    # ── Metadata ─────────────────────────────────────────────────────────────
    session_id: str
    trace_url: str                  # LangSmith trace URL
    error: str                      # Error message if any node fails


# Reducers for parallel fan-out nodes
def _merge_lists(a: list, b: list) -> list:
    return a + b


class ParallelAgentState(TypedDict):
    """State for parallel agent execution (fan-out)."""
    query: str
    expanded_query: str
    literacy_mode: str
    query_type: str
    # Parallel outputs accumulate
    research_results: Annotated[list[dict], _merge_lists]
    research_answer: str
    trial_results: Annotated[list[dict], _merge_lists]
    trial_answer: str
    drug_results: Annotated[list[dict], _merge_lists]
    drug_answer: str
    emotional_answer: str
    emotional_resources: list[dict]
