"""
NumPy In-Memory Vector Store (Version B)
Zero external dependencies. Stores embeddings in a numpy matrix on disk.
Loads into RAM on startup. Perfect for development and demo.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from gbm_copilot.config import NUMPY_INDEX_PATH
from gbm_copilot.embeddings.embedder import cosine_similarity

_EMBEDDINGS_FILE = NUMPY_INDEX_PATH / "embeddings.npy"
_METADATA_FILE = NUMPY_INDEX_PATH / "metadata.pkl"

# In-memory cache
_embeddings_matrix: np.ndarray | None = None
_metadata_list: list[dict] | None = None


def load_index() -> tuple[np.ndarray, list[dict]]:
    """Load index from disk into memory."""
    global _embeddings_matrix, _metadata_list
    if _embeddings_matrix is None:
        if _EMBEDDINGS_FILE.exists() and _METADATA_FILE.exists():
            _embeddings_matrix = np.load(str(_EMBEDDINGS_FILE))
            with open(_METADATA_FILE, "rb") as f:
                _metadata_list = pickle.load(f)
        else:
            _embeddings_matrix = np.zeros((0, 1024), dtype=np.float32)
            _metadata_list = []
    return _embeddings_matrix, _metadata_list


def save_index(embeddings: np.ndarray, metadata: list[dict]) -> None:
    """Persist index to disk."""
    global _embeddings_matrix, _metadata_list
    NUMPY_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    np.save(str(_EMBEDDINGS_FILE), embeddings)
    with open(_METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)
    _embeddings_matrix = embeddings
    _metadata_list = metadata


def upsert_chunks(chunks: list[dict], embeddings: np.ndarray) -> int:
    """
    Add chunks + embeddings to the numpy index.
    Appends to existing index if it exists.
    """
    existing_embs, existing_meta = load_index()

    # Deduplicate by source+pmid+chunk_index
    existing_keys = {
        (m.get("pmid", ""), m.get("chunk_index", 0), m.get("source", ""))
        for m in existing_meta
    }

    new_embs: list[np.ndarray] = []
    new_meta: list[dict] = []
    for chunk, emb in zip(chunks, embeddings):
        key = (chunk.get("pmid", ""), chunk.get("chunk_index", 0), chunk.get("source", ""))
        if key not in existing_keys:
            new_embs.append(emb)
            new_meta.append({
                "text": chunk["text"],
                "source": chunk.get("source", ""),
                "title": chunk.get("title", "")[:500],
                "url": chunk.get("url", ""),
                "pmid": chunk.get("pmid", ""),
                "year": str(chunk.get("year", "")),
                "chunk_index": chunk.get("chunk_index", 0),
            })

    if not new_embs:
        return 0

    if existing_embs.shape[0] > 0:
        combined_embs = np.vstack([existing_embs] + new_embs)
    else:
        combined_embs = np.array(new_embs)

    combined_meta = existing_meta + new_meta
    save_index(combined_embs, combined_meta)
    return len(new_embs)


def query_similar(
    query_embedding: np.ndarray,
    top_k: int = 50,
) -> list[dict[str, Any]]:
    """
    Query the numpy index for similar chunks using cosine similarity.
    O(n) but fast enough for <500k chunks.
    """
    embs, meta = load_index()
    if embs.shape[0] == 0:
        return []

    similarities = cosine_similarity(query_embedding, embs)
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "score": float(similarities[idx]),
            **meta[idx],
        })
    return results


def get_stats() -> dict:
    """Return index statistics."""
    embs, meta = load_index()
    return {
        "total_vectors": embs.shape[0],
        "embedding_dim": embs.shape[1] if embs.shape[0] > 0 else 0,
        "index_path": str(NUMPY_INDEX_PATH),
    }


def clear_index() -> None:
    """Clear the index (for testing/reingestion)."""
    global _embeddings_matrix, _metadata_list
    if _EMBEDDINGS_FILE.exists():
        _EMBEDDINGS_FILE.unlink()
    if _METADATA_FILE.exists():
        _METADATA_FILE.unlink()
    _embeddings_matrix = None
    _metadata_list = None
