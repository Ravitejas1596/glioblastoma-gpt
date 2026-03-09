"""
FDA Drug Labels Fetcher
Fetches GBM-relevant drug label information using the openFDA API.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

FDA_BASE = "https://api.fda.gov/drug/label.json"

GBM_DRUGS = [
    "temozolomide",
    "bevacizumab", 
    "carmustine",
    "lomustine",
    "vincristine",
    "procarbazine",
    "dexamethasone",
    "levetiracetam",
]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_drug_label(drug_name: str) -> dict[str, Any] | None:
    """
    Fetch FDA drug label for a specific drug.
    Returns structured dict with key label sections.
    """
    params = {
        "search": f'openfda.generic_name:"{drug_name}" OR openfda.brand_name:"{drug_name}"',
        "limit": 1,
    }
    async with httpx.AsyncClient() as client:
        resp = await client.get(FDA_BASE, params=params, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        return None

    label = results[0]
    openfda = label.get("openfda", {})

    return {
        "drug_name": drug_name,
        "brand_names": openfda.get("brand_name", []),
        "generic_names": openfda.get("generic_name", []),
        "manufacturer": openfda.get("manufacturer_name", [""])[0],
        "indication": _join_sections(label.get("indications_and_usage", [])),
        "mechanism": _join_sections(label.get("mechanism_of_action", [])),
        "dosage": _join_sections(label.get("dosage_and_administration", [])),
        "warnings": _join_sections(label.get("warnings", [])),
        "adverse_reactions": _join_sections(label.get("adverse_reactions", [])),
        "contraindications": _join_sections(label.get("contraindications", [])),
        "drug_interactions": _join_sections(label.get("drug_interactions", [])),
        "pharmacokinetics": _join_sections(label.get("clinical_pharmacology", [])),
        "source": "FDA drug label (openFDA)",
        "url": f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={openfda.get('application_number', [''])[0]}",
    }


def _join_sections(sections: list[str]) -> str:
    """Join multiple text sections, strip HTML tags."""
    import re
    text = " ".join(sections)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:5000]  # Cap at 5000 chars


async def fetch_all_gbm_drug_labels() -> list[dict[str, Any]]:
    """Fetch FDA labels for all GBM-relevant drugs."""
    import asyncio
    tasks = [fetch_drug_label(drug) for drug in GBM_DRUGS]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if isinstance(r, dict)]
