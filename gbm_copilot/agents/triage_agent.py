"""
Triage Agent
Classifies incoming queries into 5 categories and routes to specialized agents.
Also handles query expansion before routing.
"""
from __future__ import annotations

import json

from gbm_copilot.llm_client import get_client, get_model
from gbm_copilot.agents.state import AgentState
from gbm_copilot.retrieval.query_expander import expand

TRIAGE_SYSTEM_PROMPT = """You are a medical query triage system for GlioblastomaGPT.
Classify the user's query into exactly ONE of these categories:

- "treatment": Questions about treatments, protocols, surgery, radiation, chemotherapy
- "trial": Questions about clinical trials, experimental treatments, research studies
- "drug": Questions about specific drugs, mechanisms, side effects, dosages, interactions
- "emotional": Expressions of fear, grief, uncertainty, or requests for emotional support
- "research": Questions about biology, mechanisms, prognosis, statistics, new research findings

Output ONLY valid JSON in this format:
{"category": "treatment", "confidence": 0.92, "reasoning": "brief reason"}"""


QUERY_TYPES = {"treatment", "trial", "drug", "emotional", "research"}


async def triage_agent(state: AgentState) -> AgentState:
    """
    Triage node: classify query + expand it medically.
    Updates state with query_type and expanded_query.
    """
    query = state["query"]

    # 1. Medical query expansion (runs in parallel with classification)
    import asyncio
    expansion_task = asyncio.create_task(expand(query, rewrite=True))

    # 2. Classification
    _client = get_client()
    _model = get_model()
    response = await _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        max_tokens=150,
        temperature=0,
    )

    try:
        result = json.loads(response.choices[0].message.content)
        category = result.get("category", "research")
        if category not in QUERY_TYPES:
            category = "research"
        confidence = float(result.get("confidence", 0.8))
    except Exception:
        category = "research"
        confidence = 0.5

    # 3. Get expanded query
    expanded = await expansion_task

    return {
        **state,
        "query_type": category,
        "triage_confidence": confidence,
        "expanded_query": expanded.rewritten,
    }
