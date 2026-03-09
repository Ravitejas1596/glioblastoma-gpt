"""
GBM Ontology Loader
Loads and queries gbm_ontology.json for acronym expansion, synonym injection,
gene alias resolution, and drug name normalization.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from gbm_copilot.config import ONTOLOGY_PATH


@lru_cache(maxsize=1)
def load_ontology() -> dict[str, Any]:
    """Load the GBM ontology JSON once and cache it."""
    with open(ONTOLOGY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_acronym(term: str) -> list[str]:
    """
    Expand a medical acronym to its full terms.
    Returns list of expansions, or empty list if not found.
    """
    ontology = load_ontology()
    acronyms: dict = ontology.get("acronyms", {})
    # Case-insensitive lookup
    for key, expansions in acronyms.items():
        if key.upper() == term.upper():
            return expansions
    return []


def get_synonyms(term: str) -> list[str]:
    """
    Get synonyms for a medical term or phrase.
    Returns list of synonyms, or empty list if not found.
    """
    ontology = load_ontology()
    synonyms: dict = ontology.get("synonyms", {})
    term_lower = term.lower()
    for key, syns in synonyms.items():
        if key.lower() == term_lower:
            return syns
    return []


def get_drug_aliases(drug_name: str) -> list[str]:
    """Get all known names/aliases for a drug."""
    ontology = load_ontology()
    drugs: dict = ontology.get("drugs", {})
    drug_lower = drug_name.lower()

    for key, info in drugs.items():
        if key.lower() == drug_lower:
            aliases = info.get("aliases", [])
            brand_names = info.get("brand_names", [])
            return list(set(aliases + brand_names))

        # Check brand names and aliases too
        all_names = (
            [key]
            + info.get("aliases", [])
            + info.get("brand_names", [])
        )
        if any(n.lower() == drug_lower for n in all_names):
            return list(set([key] + info.get("aliases", []) + info.get("brand_names", [])))

    return []


def get_gene_info(gene_name: str) -> dict[str, Any] | None:
    """Get full gene information including clinical relevance."""
    ontology = load_ontology()
    genes: dict = ontology.get("genes", {})
    gene_upper = gene_name.upper()

    for key, info in genes.items():
        if key.upper() == gene_upper:
            return {"name": key, **info}

        # Check aliases
        aliases = info.get("aliases", [])
        if any(a.upper() == gene_upper for a in aliases):
            return {"name": key, **info}

    return None


def expand_query(query: str) -> tuple[str, list[str]]:
    """
    Expand a query by injecting synonyms and acronym expansions.
    Returns (expanded_query, list_of_added_terms).

    Strategy:
    1. Find acronyms in query → add expansions
    2. Find synonym keys in query → add synonyms
    3. Find drug/gene names → add aliases
    """
    ontology = load_ontology()
    added_terms: list[str] = []
    query_lower = query.lower()

    # 1. Acronym expansion
    for acronym, expansions in ontology.get("acronyms", {}).items():
        # Match whole-word acronym
        import re
        pattern = r'\b' + re.escape(acronym) + r'\b'
        if re.search(pattern, query, re.IGNORECASE):
            added_terms.extend(expansions[:3])  # Top 3 expansions

    # 2. Synonym injection
    for phrase, synonyms in ontology.get("synonyms", {}).items():
        if phrase.lower() in query_lower:
            added_terms.extend(synonyms[:2])  # Top 2 synonyms

    # 3. Gene name expansion
    for gene, info in ontology.get("genes", {}).items():
        import re
        pattern = r'\b' + re.escape(gene) + r'\b'
        if re.search(pattern, query, re.IGNORECASE):
            added_terms.append(info.get("full_name", ""))
            added_terms.extend(info.get("aliases", [])[:2])

    # 4. Drug name expansion
    for drug, info in ontology.get("drugs", {}).items():
        if drug.lower() in query_lower:
            added_terms.extend(info.get("aliases", [])[:2])
            added_terms.extend(info.get("brand_names", [])[:1])

    # Deduplicate and filter empty strings
    added_terms = list(dict.fromkeys(t for t in added_terms if t and t.lower() not in query_lower))

    if added_terms:
        expanded = query + " " + " ".join(added_terms)
    else:
        expanded = query

    return expanded, added_terms


def get_all_drug_names() -> list[str]:
    """Return all drug names and aliases for NER seeding."""
    ontology = load_ontology()
    names: list[str] = []
    for drug, info in ontology.get("drugs", {}).items():
        names.append(drug)
        names.extend(info.get("brand_names", []))
        names.extend(info.get("aliases", []))
    return list(set(names))


def get_all_gene_names() -> list[str]:
    """Return all gene names and aliases for NER seeding."""
    ontology = load_ontology()
    names: list[str] = []
    for gene, info in ontology.get("genes", {}).items():
        names.append(gene)
        names.extend(info.get("aliases", []))
    return list(set(names))
