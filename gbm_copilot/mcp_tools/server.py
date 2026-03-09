"""
FastMCP Server — GBM Copilot MCP Tools
Exposes clinical trial search and drug lookup as MCP tools.
"""
from __future__ import annotations

import asyncio
from fastmcp import FastMCP

from gbm_copilot.config import MCP_PORT

mcp = FastMCP("GBM Copilot MCP Tools")


@mcp.tool()
async def search_clinical_trials(
    condition: str = "glioblastoma",
    location: str | None = None,
    phase: str | None = None,
    max_results: int = 20,
) -> dict:
    """
    Search for recruiting GBM clinical trials on clinicaltrials.gov.
    
    Args:
        condition: Medical condition to search (default: glioblastoma)
        location: City/state to find nearby trials (e.g. "Boston, MA")
        phase: Trial phase filter (e.g. "Phase II", "Phase III")
        max_results: Maximum number of trials to return (max 50)
    
    Returns:
        Dictionary with list of clinical trials and metadata
    """
    from gbm_copilot.ingest.clinical_trials_fetcher import search_trials

    phase_map = {
        "Phase I": ["PHASE1"],
        "Phase II": ["PHASE2"],
        "Phase III": ["PHASE3"],
        "Phase I/II": ["PHASE1_PHASE2"],
        "Phase II/III": ["PHASE2_PHASE3"],
    }

    phases = phase_map.get(phase) if phase else None

    try:
        trials = await search_trials(
            condition=condition,
            location=location,
            phases=phases,
            max_results=min(max_results, 50),
        )
        return {
            "success": True,
            "count": len(trials),
            "trials": trials,
            "source": "clinicaltrials.gov",
        }
    except Exception as e:
        return {"success": False, "error": str(e), "trials": []}


@mcp.tool()
async def lookup_drug_info(
    drug_name: str,
    info_type: str = "all",
) -> dict:
    """
    Look up GBM drug information from FDA labels and medical ontology.
    
    Args:
        drug_name: Drug name (generic or brand, e.g. "temozolomide" or "Temodar")
        info_type: Type of info: "mechanism", "side_effects", "interactions", "all"
    
    Returns:
        Dictionary with drug information
    """
    from gbm_copilot.ontology.ontology_loader import load_ontology, get_drug_aliases
    from gbm_copilot.ingest.fda_fetcher import fetch_drug_label

    # Try ontology first (fast)
    ontology = load_ontology()
    drug_lower = drug_name.lower()
    ontology_info = None
    for name, info in ontology.get("drugs", {}).items():
        all_names = [name] + info.get("brand_names", []) + info.get("aliases", [])
        if any(n.lower() == drug_lower for n in all_names):
            ontology_info = {"drug_name": name, **info}
            break

    # Try FDA label
    try:
        fda_info = await fetch_drug_label(drug_name)
    except Exception:
        fda_info = None

    if not ontology_info and not fda_info:
        return {"success": False, "error": f"Drug '{drug_name}' not found", "drug_name": drug_name}

    result = {"success": True, "drug_name": drug_name}

    if ontology_info:
        result["mechanism"] = ontology_info.get("mechanism", "")
        result["indication"] = ontology_info.get("indication", "")
        result["class"] = ontology_info.get("class", "")
        result["brand_names"] = ontology_info.get("brand_names", [])
        result["aliases"] = ontology_info.get("aliases", [])

    if fda_info:
        result.update({
            "fda_mechanism": fda_info.get("mechanism", ""),
            "side_effects": fda_info.get("adverse_reactions", ""),
            "warnings": fda_info.get("warnings", ""),
            "contraindications": fda_info.get("contraindications", ""),
            "drug_interactions": fda_info.get("drug_interactions", ""),
            "manufacturer": fda_info.get("manufacturer", ""),
        })

    return result


if __name__ == "__main__":
    import uvicorn
    from fastmcp.server.http import create_sse_app
    app = create_sse_app(mcp)
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT)
