"""
Synonym Expansion Ablation Study
Compares retrieval hit-rate WITH vs WITHOUT medical synonym expansion.
Expected: +25-35% improvement on acronym-heavy queries.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

# Acronym-heavy queries where expansion should help most
ACRONYM_QUERIES = [
    {"query": "TMZ efficacy IDH wt GBM", "expected_keywords": ["temozolomide", "IDH wild-type", "glioblastoma"]},
    {"query": "MGMT methylation TMZ response", "expected_keywords": ["O6-methylguanine", "temozolomide", "predictive"]},
    {"query": "EGFR amplification treatment options", "expected_keywords": ["epidermal growth factor receptor", "glioblastoma"]},
    {"query": "TTF device phase III trial", "expected_keywords": ["tumor treating fields", "Optune", "alternating electric"]},
    {"query": "BEV recurrent GBM OS PFS", "expected_keywords": ["bevacizumab", "overall survival", "progression-free survival"]},
    {"query": "CCNU PCV IDH mutant glioma", "expected_keywords": ["lomustine", "procarbazine vincristine", "IDH mutant"]},
    {"query": "GTR vs STR GBM outcome", "expected_keywords": ["gross total resection", "subtotal resection", "extent of resection"]},
    {"query": "KPS ECOG performance status GBM", "expected_keywords": ["Karnofsky", "performance status", "functional"]},
    {"query": "BBB penetration chemotherapy CNS", "expected_keywords": ["blood-brain barrier", "central nervous system", "drug"]},
    {"query": "5-ALA fluorescence guided surgery GBM", "expected_keywords": ["aminolevulinic acid", "fluorescence", "Gliolan"]},
]


def retrieval_hit_rate(chunks: list[dict], keywords: list[str]) -> float:
    """Check what fraction of expected keywords appear in retrieved chunks."""
    if not chunks or not keywords:
        return 0.0
    combined_text = " ".join(c.get("text", "").lower() for c in chunks)
    hits = sum(1 for kw in keywords if kw.lower() in combined_text)
    return hits / len(keywords)


async def run_ablation():
    """
    Compare retrieval with and without ontology-based query expansion.
    """
    from gbm_copilot.retrieval.hybrid_retriever import retrieve
    from gbm_copilot.ontology.ontology_loader import expand_query

    results = []
    console.print("\n[bold cyan]🔬 Synonym Expansion Ablation Study[/]\n")

    for item in ACRONYM_QUERIES:
        query = item["query"]
        keywords = item["expected_keywords"]

        # Without expansion
        chunks_no_expansion = retrieve(query, top_k=10)
        hit_rate_without = retrieval_hit_rate(chunks_no_expansion, keywords)

        # With ontology expansion
        expanded_query, added_terms = expand_query(query)
        chunks_with_expansion = retrieve(expanded_query, top_k=10)
        hit_rate_with = retrieval_hit_rate(chunks_with_expansion, keywords)

        improvement = hit_rate_with - hit_rate_without
        pct_improvement = (improvement / max(hit_rate_without, 0.01)) * 100

        results.append({
            "query": query,
            "expanded_query": expanded_query,
            "added_terms_count": len(added_terms),
            "hit_rate_without": hit_rate_without,
            "hit_rate_with": hit_rate_with,
            "absolute_improvement": improvement,
            "pct_improvement": pct_improvement,
        })

        console.print(
            f"[cyan]{query[:45]}[/]\n"
            f"  Without: {hit_rate_without:.1%} → With: {hit_rate_with:.1%} "
            f"({'[green]+' if improvement >= 0 else '[red]'}{pct_improvement:+.1f}%[/])"
        )

    # Summary
    avg_without = sum(r["hit_rate_without"] for r in results) / len(results)
    avg_with = sum(r["hit_rate_with"] for r in results) / len(results)
    avg_improvement = (avg_with - avg_without) / max(avg_without, 0.01) * 100

    table = Table(title="Ablation Summary", style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")
    table.add_row("Avg Hit Rate (No Expansion)", f"{avg_without:.1%}")
    table.add_row("Avg Hit Rate (With Expansion)", f"{avg_with:.1%}")
    table.add_row("Avg Improvement", f"{avg_improvement:+.1f}%")
    table.add_row("Queries Tested", str(len(results)))
    console.print(table)

    # Save results
    with open("ablation_results.json", "w") as f:
        json.dump({"results": results, "summary": {
            "avg_without": avg_without,
            "avg_with": avg_with,
            "avg_pct_improvement": avg_improvement,
        }}, f, indent=2)

    console.print("\n[green]Results saved to ablation_results.json[/]")
    return results


if __name__ == "__main__":
    asyncio.run(run_ablation())
