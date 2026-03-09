"""
ClinicalTrials.gov API Fetcher
Fetches recruiting GBM clinical trials using the v2 ClinicalTrials.gov API.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

CT_BASE = "https://clinicaltrials.gov/api/v2"

GBM_CONDITIONS = ["glioblastoma", "glioblastoma multiforme", "GBM", "grade IV glioma"]
RELEVANT_PHASES = ["PHASE2", "PHASE3", "PHASE1_PHASE2", "PHASE2_PHASE3"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def search_trials(
    condition: str = "glioblastoma",
    status: str = "RECRUITING",
    phases: list[str] | None = None,
    location: str | None = None,
    max_distance_miles: int = 200,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """
    Search clinicaltrials.gov for GBM trials.
    Returns list of structured trial dicts with plain-English summaries.
    """
    if phases is None:
        phases = RELEVANT_PHASES

    params: dict[str, Any] = {
        "query.cond": condition,
        "filter.overallStatus": status,
        "filter.phase": "|".join(phases),
        "pageSize": min(max_results, 100),
        "format": "json",
        "fields": "|".join([
            "NCTId", "BriefTitle", "OfficialTitle", "BriefSummary",
            "OverallStatus", "Phase", "StudyType",
            "PrimaryOutcomeMeasure", "InterventionName", "InterventionType",
            "EligibilityCriteria", "MinimumAge", "MaximumAge",
            "LocationFacility", "LocationCity", "LocationState", "LocationZip",
            "LocationCountry", "ContactName", "ContactPhone", "ContactEMail",
            "StartDate", "PrimaryCompletionDate", "CompletionDate",
            "EnrollmentCount", "LeadSponsorName",
        ]),
    }

    if location:
        params["filter.geo"] = f"distance({location},{max_distance_miles}mi)"

    headers = {
        "User-Agent": "GlioblastomaGPT/1.0 (https://github.com/gbm-copilot; ravitejas1596@gmail.com)",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(headers=headers) as client:
        resp = await client.get(f"{CT_BASE}/studies", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

    studies = data.get("studies", [])
    return [_parse_trial(s) for s in studies]


def _parse_trial(study: dict) -> dict[str, Any]:
    """Parse a raw ClinicalTrials.gov study into a clean structured dict."""
    proto = study.get("protocolSection", {})
    id_module = proto.get("identificationModule", {})
    status_module = proto.get("statusModule", {})
    desc_module = proto.get("descriptionModule", {})
    design_module = proto.get("designModule", {})
    eligibility_module = proto.get("eligibilityModule", {})
    contacts_module = proto.get("contactsLocationsModule", {})
    interventions = proto.get("armsInterventionsModule", {}).get("interventions", [])
    sponsor_module = proto.get("sponsorCollaboratorsModule", {})

    # Extract locations
    locations = []
    for loc in contacts_module.get("locations", [])[:10]:  # First 10 locations
        locations.append({
            "facility": loc.get("facility", ""),
            "city": loc.get("city", ""),
            "state": loc.get("state", ""),
            "country": loc.get("country", ""),
            "zip": loc.get("zip", ""),
        })

    return {
        "nct_id": id_module.get("nctId", ""),
        "title": id_module.get("briefTitle", ""),
        "official_title": id_module.get("officialTitle", ""),
        "summary": desc_module.get("briefSummary", "").strip(),
        "status": status_module.get("overallStatus", ""),
        "phase": _format_phase(design_module.get("phases", [])),
        "study_type": design_module.get("studyType", ""),
        "interventions": [
            {"name": i.get("name", ""), "type": i.get("type", "")}
            for i in interventions[:5]
        ],
        "eligibility": eligibility_module.get("eligibilityCriteria", "")[:2000],
        "min_age": eligibility_module.get("minimumAge", ""),
        "max_age": eligibility_module.get("maximumAge", ""),
        "locations": locations,
        "start_date": status_module.get("startDateStruct", {}).get("date", ""),
        "completion_date": status_module.get("primaryCompletionDateStruct", {}).get("date", ""),
        "enrollment": design_module.get("enrollmentInfo", {}).get("count", ""),
        "sponsor": sponsor_module.get("leadSponsor", {}).get("name", ""),
        "url": f"https://clinicaltrials.gov/study/{id_module.get('nctId', '')}",
        "source": "clinicaltrials.gov",
    }


def _format_phase(phases: list[str]) -> str:
    phase_map = {
        "PHASE1": "Phase I",
        "PHASE2": "Phase II",
        "PHASE3": "Phase III",
        "PHASE4": "Phase IV",
        "PHASE1_PHASE2": "Phase I/II",
        "PHASE2_PHASE3": "Phase II/III",
        "NA": "N/A",
    }
    return " / ".join(phase_map.get(p, p) for p in phases) if phases else "Unknown"


async def fetch_all_gbm_trials() -> list[dict[str, Any]]:
    """Fetch all recruiting GBM trials for ingestion into knowledge base."""
    all_trials: list[dict] = []
    for condition in GBM_CONDITIONS[:2]:  # GBM + glioblastoma to avoid redundancy
        trials = await search_trials(condition=condition, max_results=100)
        seen = {t["nct_id"] for t in all_trials}
        all_trials.extend(t for t in trials if t["nct_id"] not in seen)
    return all_trials
