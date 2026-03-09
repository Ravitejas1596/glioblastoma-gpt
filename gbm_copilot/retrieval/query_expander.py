"""
Medical Query Expander
Pipeline: user query → scispaCy NER → acronym expansion → synonym injection → GPT-4o rewrite

This is the key insight that makes our RAG actually work for medical text.
A patient asking "will chemo help?" gets expanded to a precise medical query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from gbm_copilot.llm_client import get_client, get_model
from gbm_copilot.ontology.ontology_loader import expand_query as ontology_expand

# Try to load scispaCy — graceful fallback if not installed
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_sci_lg")
        except Exception:
            try:
                import spacy
                _nlp = spacy.load("en_core_web_sm")  # Fallback to basic model
            except Exception:
                _nlp = False  # Signal unavailable
    return _nlp if _nlp is not False else None


@dataclass
class ExpandedQuery:
    original: str
    ontology_expanded: str
    ner_entities: list[str]
    added_terms: list[str]
    rewritten: str  # GPT-4o rewrite for the final retrieval query


REWRITE_SYSTEM_PROMPT = """You are a medical search query optimizer for a glioblastoma (GBM) research system.
Your task: rewrite a patient/caregiver query into a precise biomedical search query.

Rules:
- Expand patient language to medical terminology (e.g., "brain cancer" → "glioblastoma")
- Include relevant gene names, drug names, and clinical terms
- Keep it to 1-3 focused search phrases
- Include both common names and technical terms (e.g., "temozolomide (TMZ, Temodar)")
- Do NOT add clinical advice or answer the question — just reformulate as a search query
- Output ONLY the reformulated search query, nothing else"""


async def expand(query: str, rewrite: bool = True) -> ExpandedQuery:
    """
    Full query expansion pipeline.
    
    1. Ontology-based acronym/synonym expansion
    2. scispaCy NER entity extraction  
    3. GPT-4o query rewrite (optional, costs tokens)
    """
    # Step 1: Ontology expansion
    ontology_expanded, added_terms = ontology_expand(query)

    # Step 2: NER entity extraction
    ner_entities: list[str] = []
    nlp = _get_nlp()
    if nlp:
        doc = nlp(query)
        ner_entities = list(set(
            ent.text for ent in doc.ents
            if ent.label_ in {"DISEASE", "CHEMICAL", "GENE_OR_GENE_PRODUCT", "CELL_TYPE"}
        ))

    # Step 3: GPT-4o rewrite
    if rewrite:
        rewritten = await _rewrite_query(query, ontology_expanded, ner_entities)
    else:
        rewritten = ontology_expanded

    return ExpandedQuery(
        original=query,
        ontology_expanded=ontology_expanded,
        ner_entities=ner_entities,
        added_terms=added_terms,
        rewritten=rewritten,
    )


async def _rewrite_query(
    original: str,
    ontology_expanded: str,
    ner_entities: list[str],
) -> str:
    """Use GPT-4o to produce a precision medical search query."""
    context_parts = [f"Original query: {original}"]
    if ontology_expanded != original:
        context_parts.append(f"Expanded terms: {ontology_expanded}")
    if ner_entities:
        context_parts.append(f"Detected medical entities: {', '.join(ner_entities)}")

    user_message = "\n".join(context_parts)

    _client = get_client()
    _model = get_model()
    response = await _client.chat.completions.create(
        model=_model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=200,
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def expand_sync(query: str) -> str:
    """
    Synchronous query expansion (ontology only, no GPT-4o).
    For use in hot paths where async isn't available.
    """
    expanded, _ = ontology_expand(query)
    return expanded
