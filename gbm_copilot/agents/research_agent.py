"""
Research Agent
Searches PubMed abstracts, NCCN/ASCO clinical guidelines, and local knowledge base.
Uses hybrid retrieval (BM25 + dense). Translates findings to the user's literacy mode.
"""
from __future__ import annotations

from gbm_copilot.llm_client import get_client, get_model
from gbm_copilot.config import LITERACY_MODE_DESCRIPTIONS
from gbm_copilot.agents.state import AgentState
from gbm_copilot.retrieval.hybrid_retriever import retrieve

RESEARCH_SYSTEM_PROMPT = """You are a GBM (glioblastoma) research specialist with expertise in neuro-oncology.
You have access to PubMed abstracts, clinical guidelines (NCCN, ASCO), and the latest GBM research.

Your task: Answer the user's question using ONLY the provided context chunks. 

Rules:
- Cite sources using [Source: title, year] format at end of relevant sentences
- Literacy mode: {literacy_mode} → {literacy_description}
- If context is insufficient, say "The available research on this specific question is limited"
- Do NOT make up statistics or cite studies not in the context
- For treatment questions, always note that individual cases vary and oncologist input is essential
- Be specific: mention gene names, drug names, trial names when relevant"""


async def research_agent(state: AgentState) -> dict:
    """
    Research agent node in LangGraph.
    Retrieves relevant chunks and generates a research-backed answer.
    """
    query = state.get("expanded_query", state["query"])
    literacy = state.get("literacy_mode", "patient")
    literacy_desc = LITERACY_MODE_DESCRIPTIONS.get(literacy, "")

    # Hybrid retrieval
    chunks = retrieve(query, top_k=8)
    state_update = {"research_results": chunks}

    if not chunks:
        return {**state_update, "research_answer": ""}

    # Build context
    context = "\n\n---\n\n".join([
        f"[{i+1}] {c.get('title', 'Research excerpt')} ({c.get('year', 'n.d.')})\n{c['text']}"
        for i, c in enumerate(chunks)
    ])

    system = RESEARCH_SYSTEM_PROMPT.format(
        literacy_mode=literacy,
        literacy_description=literacy_desc,
    )

    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {state['query']}"
        },
    ]

    # Include conversation history for multi-turn
    if state.get("conversation_history"):
        history = state["conversation_history"][-4:]  # Last 4 turns
        messages = [messages[0]] + history + [messages[-1]]

    _client = get_client()
    _model = get_model()
    response = await _client.chat.completions.create(
        model=_model,
        messages=messages,
        max_tokens=1000,
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return {**state_update, "research_answer": answer}
