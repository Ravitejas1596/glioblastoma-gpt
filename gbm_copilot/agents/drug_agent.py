"""
Drug Info Agent
Retrieves drug information from FDA labels + curated GBM ontology.
Explains mechanisms, side effects, and interactions in patient-appropriate language.
"""
from __future__ import annotations

from gbm_copilot.llm_client import get_client, get_model
from gbm_copilot.config import LITERACY_MODE_DESCRIPTIONS
from gbm_copilot.agents.state import AgentState
from gbm_copilot.ingest.fda_fetcher import fetch_drug_label
from gbm_copilot.ontology.ontology_loader import get_drug_aliases, load_ontology

DRUG_SYSTEM_PROMPT = """You are a GBM (glioblastoma) pharmacology specialist.
Explain the drug information in a way appropriate for: {literacy_mode} level.
{literacy_description}

Important:
- Never give specific dosage advice — always refer to the oncologist/pharmacist
- Explain side effects clearly so patients know what to watch for
- Mention drug interactions that are most relevant for GBM patients
- Use both the brand name and generic name when first mentioning a drug
- End with: "Always discuss this medication with your oncology team."
"""


async def drug_agent(state: AgentState) -> dict:
    """Drug info agent: FDA labels + ontology + GPT-4o explanation."""
    query = state["query"]
    literacy = state.get("literacy_mode", "patient")

    # Extract drug names from query using ontology
    drug_info_list = _find_drugs_in_query(query)
    drug_results = []

    # Fetch FDA labels
    import asyncio
    tasks = [fetch_drug_label(drug) for drug in drug_info_list[:3]]
    fda_results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in fda_results:
        if isinstance(result, dict):
            drug_results.append(result)

    # Also search knowledge base for drug info
    from gbm_copilot.retrieval.hybrid_retriever import retrieve
    expanded_query = state.get("expanded_query", query) + " drug mechanism side effects"
    kb_chunks = retrieve(expanded_query, top_k=4)
    drug_results_kb = kb_chunks

    state_update = {"drug_results": drug_results + drug_results_kb}

    # Build context
    context_parts = []
    for drug in drug_results:
        context_parts.append(
            f"Drug: {drug['drug_name'].title()} (Brand names: {', '.join(drug.get('brand_names', []))})\n"
            f"Mechanism: {drug.get('mechanism', 'Not available')[:500]}\n"
            f"Side effects: {drug.get('adverse_reactions', 'See package insert')[:500]}\n"
            f"Warnings: {drug.get('warnings', '')[:300]}"
        )

    if not context_parts and not drug_results_kb:
        # Fallback to ontology
        ontology = load_ontology()
        drugs = ontology.get("drugs", {})
        for drug_name_q in drug_info_list:
            if drug_name_q.lower() in drugs:
                info = drugs[drug_name_q.lower()]
                context_parts.append(
                    f"Drug: {drug_name_q}\n"
                    f"Mechanism: {info.get('mechanism', 'N/A')}\n"
                    f"Indication: {info.get('indication', 'N/A')}"
                )

    context = "\n\n".join(context_parts)
    if drug_results_kb:
        context += "\n\n" + "\n\n".join(c["text"][:400] for c in drug_results_kb)

    if not context.strip():
        return {**state_update, "drug_answer": ""}

    _client = get_client()
    _model = get_model()
    response = await _client.chat.completions.create(
        model=_model,
        messages=[
            {
                "role": "system",
                "content": DRUG_SYSTEM_PROMPT.format(
                    literacy_mode=literacy,
                    literacy_description=LITERACY_MODE_DESCRIPTIONS.get(literacy, ""),
                )
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nDrug information:\n{context}"
            },
        ],
        max_tokens=800,
        temperature=0.2,
    )

    return {**state_update, "drug_answer": response.choices[0].message.content.strip()}


def _find_drugs_in_query(query: str) -> list[str]:
    """Find drug names mentioned in the query using the ontology."""
    query_lower = query.lower()
    ontology = load_ontology()
    found: list[str] = []
    for drug_name, info in ontology.get("drugs", {}).items():
        all_names = [drug_name] + info.get("brand_names", []) + info.get("aliases", [])
        if any(n.lower() in query_lower for n in all_names):
            found.append(drug_name)
    return list(set(found))
