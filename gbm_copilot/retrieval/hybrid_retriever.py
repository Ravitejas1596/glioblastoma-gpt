"""
Hybrid Retriever
Combines BM25 (keyword) + Dense (BGE-M3) retrieval using
Reciprocal Rank Fusion (RRF) for optimal medical retrieval.

Key insight: BM25 catches exact drug/gene/acronym matches,
dense catches semantic similarity. RRF merges both gracefully.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from gbm_copilot.config import RETRIEVAL_MODE
from gbm_copilot.retrieval import bm25_retriever
from gbm_copilot.embeddings import embedder

RRF_K = 60  # Standard RRF constant


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a document at given rank."""
    return 1.0 / (k + rank + 1)


def _rrf_merge(
    bm25_results: list[dict],
    dense_results: list[dict],
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> list[dict[str, Any]]:
    """
    Merge BM25 and dense results using weighted RRF.
    Dense gets slightly higher weight for semantic understanding,
    but BM25 is critical for medical terminology.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}

    # Key function — use url+chunk_index or title as unique id
    def doc_key(d: dict) -> str:
        return f"{d.get('url', '')}__{d.get('chunk_index', 0)}__{d.get('source', '')}"

    for rank, doc in enumerate(bm25_results):
        key = doc_key(doc)
        scores[key] = scores.get(key, 0.0) + bm25_weight * _rrf_score(rank)
        docs[key] = doc

    for rank, doc in enumerate(dense_results):
        key = doc_key(doc)
        scores[key] = scores.get(key, 0.0) + dense_weight * _rrf_score(rank)
        if key not in docs:
            docs[key] = doc

    # Sort by combined RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    results = []
    for key in sorted_keys:
        doc = dict(docs[key])
        doc["hybrid_score"] = scores[key]
        results.append(doc)

    return results


def retrieve(
    query: str,
    query_embedding: np.ndarray | None = None,
    top_k: int = 10,
    bm25_top_k: int = 50,
    dense_top_k: int = 50,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: BM25 + dense → RRF fusion → top_k results.

    Args:
        query: The search query (already expanded by query_expander)
        query_embedding: Pre-computed embedding (optional, computed if None)
        top_k: Number of final results to return
        bm25_top_k: Candidate pool from BM25
        dense_top_k: Candidate pool from dense retrieval
    """
    # BM25 retrieval
    bm25_results = bm25_retriever.search(query, top_k=bm25_top_k)

    # Dense retrieval
    if query_embedding is None:
        query_embedding = embedder.embed_query(query)

    if RETRIEVAL_MODE == "pinecone":
        from gbm_copilot.embeddings import pinecone_store
        dense_results = pinecone_store.query_similar(query_embedding, top_k=dense_top_k)
    else:
        from gbm_copilot.embeddings import numpy_store
        dense_results = numpy_store.query_similar(query_embedding, top_k=dense_top_k)

    # Fuse with RRF
    merged = _rrf_merge(bm25_results, dense_results)
    return merged[:top_k]


def retrieve_with_scores(
    query: str,
    top_k: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Retrieve with detailed scoring metadata for debugging/evaluation.
    Returns (results, score_breakdown).
    """
    from gbm_copilot.embeddings import embedder as emb_module
    query_embedding = emb_module.embed_query(query)

    bm25_results = bm25_retriever.search(query, top_k=50)
    if RETRIEVAL_MODE == "pinecone":
        from gbm_copilot.embeddings import pinecone_store
        dense_results = pinecone_store.query_similar(query_embedding, top_k=50)
    else:
        from gbm_copilot.embeddings import numpy_store
        dense_results = numpy_store.query_similar(query_embedding, top_k=50)

    merged = _rrf_merge(bm25_results, dense_results)

    score_breakdown = {
        "bm25_candidates": len(bm25_results),
        "dense_candidates": len(dense_results),
        "merged_candidates": len(merged),
        "retrieval_mode": RETRIEVAL_MODE,
    }

    return merged[:top_k], score_breakdown
