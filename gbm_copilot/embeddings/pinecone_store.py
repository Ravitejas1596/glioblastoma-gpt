"""
Pinecone Vector Store (Version A)
Upserts and queries embeddings using Pinecone serverless.
"""
from __future__ import annotations

import uuid
from typing import Any

import numpy as np

from gbm_copilot.config import (
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_ENVIRONMENT
)

_index = None


def _get_index():
    global _index
    if _index is None:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=PINECONE_API_KEY)

        existing = [idx.name for idx in pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1024,  # BGE-M3
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        _index = pc.Index(PINECONE_INDEX_NAME)
    return _index


def upsert_chunks(chunks: list[dict], embeddings: np.ndarray, batch_size: int = 100) -> int:
    """
    Upsert chunks + embeddings to Pinecone.
    Returns count of vectors upserted.
    """
    index = _get_index()
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = chunk.get("id") or str(uuid.uuid4())
        vectors.append({
            "id": chunk_id,
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk["text"][:1000],  # Pinecone metadata limit
                "source": chunk.get("source", ""),
                "title": chunk.get("title", "")[:200],
                "url": chunk.get("url", ""),
                "pmid": chunk.get("pmid", ""),
                "year": str(chunk.get("year", "")),
                "chunk_index": chunk.get("chunk_index", 0),
            },
        })

    total = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        total += len(batch)
    return total


def query_similar(
    query_embedding: np.ndarray,
    top_k: int = 50,
    filter_dict: dict | None = None,
) -> list[dict[str, Any]]:
    """Query Pinecone for similar chunks."""
    index = _get_index()
    kwargs: dict[str, Any] = {
        "vector": query_embedding.tolist(),
        "top_k": top_k,
        "include_metadata": True,
    }
    if filter_dict:
        kwargs["filter"] = filter_dict

    response = index.query(**kwargs)
    results = []
    for match in response.matches:
        results.append({
            "score": match.score,
            "text": match.metadata.get("text", ""),
            "source": match.metadata.get("source", ""),
            "title": match.metadata.get("title", ""),
            "url": match.metadata.get("url", ""),
            "pmid": match.metadata.get("pmid", ""),
            "year": match.metadata.get("year", ""),
        })
    return results


def get_stats() -> dict:
    """Return Pinecone index statistics."""
    return _get_index().describe_index_stats().to_dict()
