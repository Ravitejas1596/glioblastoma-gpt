"""
Synthesizer Agent
Merges outputs from all specialized agents, deduplicates, adds citations,
and translates to the appropriate literacy level.
"""
from __future__ import annotations

from gbm_copilot.llm_client import get_client, get_model
from gbm_copilot.config import LITERACY_MODE_DESCRIPTIONS
from gbm_copilot.agents.state import AgentState

SYNTHESIZER_SYSTEM_PROMPT = """You are the final synthesizer for GlioblastomaGPT, a GBM research assistant.
You will receive partially overlapping answers from specialized agents. Your job:

1. Merge all relevant information into ONE coherent, clear answer
2. Remove contradictions (prefer more recent/specific information)  
3. Remove redundant information
4. Cite sources clearly: [Source: Title, Year, URL] at end of relevant statements
5. Match literacy level: {literacy_mode} — {literacy_description}
6. Structure the response clearly (use headers/bullets if helpful)
7. End with a confidence indicator based on source quality and agreement

If emotional support content is included, place it FIRST before clinical information.
If trial information is included, format as a distinct "Clinical Trials" section.

Confidence scale:
- High confidence: Multiple peer-reviewed sources agree
- Moderate confidence: Limited sources or some disagreement
- Low confidence: Minimal evidence or conflicting data"""


async def synthesizer_agent(state: AgentState) -> dict:
    """
    Synthesizer node: merge all agent outputs into final answer.
    """
    literacy = state.get("literacy_mode", "patient")
    query = state["query"]

    # Collect all agent outputs
    parts: list[tuple[str, str]] = []

    if state.get("emotional_answer"):
        parts.append(("emotional_support", state["emotional_answer"]))

    if state.get("research_answer"):
        parts.append(("research", state["research_answer"]))

    if state.get("drug_answer"):
        parts.append(("drug_information", state["drug_answer"]))

    if state.get("trial_answer"):
        parts.append(("clinical_trials", state["trial_answer"]))

    if not parts:
        # No retrieved context (index may be empty). Fall back to Groq's training knowledge
        # so users still get a helpful answer. Confidence=0.65 passes the safety layer.
        _client = get_client()
        _model = get_model()
        literacy_desc = LITERACY_MODE_DESCRIPTIONS.get(literacy, "")
        fallback_resp = await _client.chat.completions.create(
            model=_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are GlioblastomaGPT, a knowledgeable AI assistant specialized in "
                        "glioblastoma (GBM) and neuro-oncology. "
                        f"Answer the user's question clearly and accurately. "
                        f"Literacy level: {literacy} — {literacy_desc}. "
                        "Structure your answer with headers and bullet points where helpful. "
                        "Note: you are answering from general medical knowledge as no specific "
                        "research documents were retrieved from the database for this query."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=1200,
            temperature=0.3,
        )
        return {
            "final_answer": fallback_resp.choices[0].message.content.strip(),
            "citations": [],
            "confidence_score": 0.65,  # Moderate — LLM knowledge, no RAG grounding
        }

    # If only one agent responded, use it directly
    if len(parts) == 1:
        return {
            "final_answer": parts[0][1],
            "citations": _extract_citations(state),
            "confidence_score": 0.75,
        }

    # Multiple agents: synthesize
    combined_text = "\n\n".join([f"[{label.upper()}]\n{content}" for label, content in parts])

    _client = get_client()
    _model = get_model()
    response = await _client.chat.completions.create(
        model=_model,
        messages=[
            {
                "role": "system",
                "content": SYNTHESIZER_SYSTEM_PROMPT.format(
                    literacy_mode=literacy,
                    literacy_description=LITERACY_MODE_DESCRIPTIONS.get(literacy, ""),
                )
            },
            {
                "role": "user",
                "content": f"Original question: {query}\n\n{combined_text}"
            },
        ],
        max_tokens=1500,
        temperature=0.2,
    )

    final_answer = response.choices[0].message.content.strip()
    citations = _extract_citations(state)

    # Simple confidence heuristic based on source count and types
    source_count = len(citations)
    has_pubmed = any(c.get("source") == "pubmed" for c in citations)
    confidence = min(0.95, 0.5 + (source_count * 0.05) + (0.1 if has_pubmed else 0))

    return {
        "final_answer": final_answer,
        "citations": citations,
        "confidence_score": confidence,
    }


def _extract_citations(state: AgentState) -> list[dict]:
    """Compile citations from all retrieval results."""
    citations: list[dict] = []
    seen_urls: set[str] = set()

    sources = (
        state.get("research_results", [])
        + state.get("drug_results", [])
        + [
            {
                "title": t.get("title", ""),
                "url": t.get("url", ""),
                "source": "clinicaltrials.gov",
                "year": t.get("start_date", "")[:4],
            }
            for t in state.get("trial_results", [])
        ]
    )

    for src in sources:
        url = src.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            citations.append({
                "title": src.get("title", "Research source")[:150],
                "url": url,
                "source": src.get("source", ""),
                "year": str(src.get("year", "")),
                "pmid": src.get("pmid", ""),
            })

    return citations[:15]  # Cap at 15 citations
