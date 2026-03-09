"""
Full Ingest Pipeline Orchestrator
Runs all data fetchers → chunker → contextual retrieval → embedder → vector store.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from gbm_copilot.config import (
    INGEST_QUICK_MODE, CHUNKS_PATH, RETRIEVAL_MODE
)
from gbm_copilot.ingest.pubmed_fetcher import fetch_pubmed_abstracts
from gbm_copilot.ingest.clinical_trials_fetcher import fetch_all_gbm_trials
from gbm_copilot.ingest.fda_fetcher import fetch_all_gbm_drug_labels
from gbm_copilot.ingest.chunker import chunk_text
from gbm_copilot.ingest.contextual_retrieval import add_context_batch
from gbm_copilot.embeddings.embedder import embed_texts
from gbm_copilot.retrieval.bm25_retriever import build_index as build_bm25

console = Console()

EMBED_BATCH_SIZE = 32  # Process embeddings in batches


async def run_ingest(quick_mode: bool | None = None):
    """
    Full ingest pipeline. Runs all sources in sequence.
    quick_mode=True: 500 PubMed abstracts (for dev/testing)
    quick_mode=False: Full 50k abstracts
    """
    if quick_mode is None:
        quick_mode = INGEST_QUICK_MODE

    start_time = time.time()
    all_chunks: list[dict] = []

    console.print(f"\n[bold cyan]🧠 GlioblastomaGPT Ingest Pipeline[/]\n"
                  f"Mode: {'[yellow]Quick (500 abstracts)[/]' if quick_mode else '[green]Full (50k abstracts)[/]'}\n"
                  f"Retrieval: [bold]{RETRIEVAL_MODE.upper()}[/]\n")

    # ── 1. PubMed ────────────────────────────────────────────────────────────
    console.print("[bold]📚 Fetching PubMed abstracts...[/]")
    pubmed_chunks: list[dict] = []
    pubmed_count = 0

    async for article in fetch_pubmed_abstracts():
        if article.get("abstract"):
            chunks = chunk_text(
                text=article["abstract"],
                metadata={
                    "title": article.get("title", ""),
                    "source": "pubmed",
                    "pmid": article.get("pmid", ""),
                    "year": article.get("year", ""),
                    "url": article.get("url", ""),
                    "journal": article.get("journal", ""),
                },
            )
            pubmed_chunks.extend(chunks)
            pubmed_count += 1
            if pubmed_count % 100 == 0:
                console.print(f"  → {pubmed_count} abstracts → {len(pubmed_chunks)} chunks")

    console.print(f"  [green]✓[/] {pubmed_count} abstracts → {len(pubmed_chunks)} chunks")
    all_chunks.extend(pubmed_chunks)

    # ── Save PubMed progress immediately (resilience) ────────────────────────
    console.print("[bold]💾 Saving PubMed chunks to disk (checkpoint)...[/]")
    with open(CHUNKS_PATH, "w") as f:
        for chunk in pubmed_chunks:
            f.write(json.dumps(chunk) + "\n")
    console.print(f"  [green]✓[/] Checkpoint saved: {len(pubmed_chunks)} PubMed chunks")

    # ── 2. Clinical Trials ───────────────────────────────────────────────────
    console.print("[bold]🔬 Fetching clinical trials...[/]")
    trial_chunks: list[dict] = []
    try:
        trials = await fetch_all_gbm_trials()
        for trial in trials:
            trial_text = (
                f"{trial['title']}\n"
                f"Phase: {trial['phase']} | Status: {trial['status']}\n"
                f"Sponsor: {trial['sponsor']}\n"
                f"Summary: {trial['summary']}\n"
                f"Eligibility: {trial.get('eligibility', '')[:500]}"
            )
            chunks = chunk_text(
                text=trial_text,
                metadata={
                    "title": trial["title"],
                    "source": "clinicaltrials.gov",
                    "url": trial["url"],
                    "year": trial.get("start_date", "")[:4],
                    "nct_id": trial["nct_id"],
                },
            )
            trial_chunks.extend(chunks)
        console.print(f"  [green]✓[/] {len(trials)} trials → {len(trial_chunks)} chunks")
    except Exception as e:
        console.print(f"  [yellow]⚠ ClinicalTrials.gov fetch failed (continuing): {e}[/]")
        trials = []
    all_chunks.extend(trial_chunks)

    # ── 3. FDA Drug Labels ───────────────────────────────────────────────────
    console.print("[bold]💊 Fetching FDA drug labels...[/]")
    drug_chunks: list[dict] = []
    try:
        drug_labels = await fetch_all_gbm_drug_labels()
        for drug in drug_labels:
            drug_text = (
                f"Drug: {drug['drug_name'].title()}\n"
                f"Brand names: {', '.join(drug.get('brand_names', []))}\n"
                f"Indication: {drug.get('indication', '')[:500]}\n"
                f"Mechanism: {drug.get('mechanism', '')[:500]}\n"
                f"Side effects: {drug.get('adverse_reactions', '')[:500]}\n"
                f"Warnings: {drug.get('warnings', '')[:300]}"
            )
            chunks = chunk_text(
                text=drug_text,
                metadata={
                    "title": f"{drug['drug_name'].title()} — FDA Drug Label",
                    "source": "fda_label",
                    "url": drug.get("url", ""),
                },
            )
            drug_chunks.extend(chunks)
        console.print(f"  [green]✓[/] {len(drug_labels)} drug labels → {len(drug_chunks)} chunks")
    except Exception as e:
        console.print(f"  [yellow]⚠ FDA fetch failed (continuing): {e}[/]")
        drug_labels = []
    all_chunks.extend(drug_chunks)

    # ── 4. Contextual Retrieval (Anthropic-style) ────────────────────────────
    console.print(f"[bold]🔄 Adding contextual retrieval to {len(all_chunks)} chunks...[/]")
    console.print("  [dim](This calls GPT-4o for each chunk — may take a while for full mode)[/]")

    # For quick mode: skip contextual retrieval on PubMed (too many API calls)
    # Only apply to drug labels and trials (higher quality return on investment)
    if quick_mode:
        enriched_chunks = all_chunks  # Skip in quick mode
    else:
        # Apply context enrichment to non-PubMed chunks for full mode
        important_chunks = [c for c in all_chunks if c.get("source") != "pubmed"]
        pubmed_only = [c for c in all_chunks if c.get("source") == "pubmed"]

        if important_chunks:
            enriched_important = await add_context_batch(important_chunks, document_text=" ")
            enriched_chunks = pubmed_only + enriched_important
        else:
            enriched_chunks = all_chunks

    console.print(f"  [green]✓[/] Contextual retrieval complete")

    # ── 5. Persist chunks to JSONL ───────────────────────────────────────────
    console.print(f"[bold]💾 Saving {len(enriched_chunks)} chunks to disk...[/]")
    with open(CHUNKS_PATH, "w") as f:
        for chunk in enriched_chunks:
            f.write(json.dumps(chunk) + "\n")
    console.print(f"  [green]✓[/] Saved to {CHUNKS_PATH}")

    # ── 6. BM25 Index ────────────────────────────────────────────────────────
    console.print("[bold]🔑 Building BM25 keyword index...[/]")
    build_bm25(enriched_chunks)
    console.print(f"  [green]✓[/] BM25 index: {len(enriched_chunks)} documents")

    # ── 7. Dense Embeddings + Vector Store ───────────────────────────────────
    console.print(f"[bold]🧬 Computing BGE-M3 embeddings ({len(enriched_chunks)} chunks)...[/]")

    texts = [c["text"] for c in enriched_chunks]
    all_embeddings = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i+EMBED_BATCH_SIZE]
        batch_embs = embed_texts(batch)
        all_embeddings.append(batch_embs)
        if (i // EMBED_BATCH_SIZE) % 10 == 0:
            console.print(f"  → {i}/{len(texts)} embedded")

    import numpy as np
    embeddings_matrix = np.vstack(all_embeddings)

    if RETRIEVAL_MODE == "pinecone":
        from gbm_copilot.embeddings.pinecone_store import upsert_chunks
    else:
        from gbm_copilot.embeddings.numpy_store import upsert_chunks

    count = upsert_chunks(enriched_chunks, embeddings_matrix)
    console.print(f"  [green]✓[/] {count} vectors added to {RETRIEVAL_MODE.upper()} store")

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]✅ Ingest complete![/] {len(enriched_chunks)} chunks in {elapsed:.1f}s\n")

    return {
        "chunks": len(enriched_chunks),
        "pubmed_articles": pubmed_count,
        "trials": len(trials),
        "drug_labels": len(drug_labels),
        "elapsed_seconds": round(elapsed, 1),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GBM Copilot Ingest Pipeline")
    parser.add_argument("--quick", action="store_true", help="Quick mode: 500 PubMed abstracts")
    args = parser.parse_args()

    asyncio.run(run_ingest(quick_mode=args.quick))
