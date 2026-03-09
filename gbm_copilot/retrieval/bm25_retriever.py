"""
BM25 Retriever
Keyword-based retrieval critical for medical text:
- exact drug names (temozolomide vs Temodar)
- gene names (MGMT, IDH1, EGFR)
- acronyms (GBM, TMZ, CCNU)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from gbm_copilot.config import BM25_INDEX_PATH

_bm25: BM25Okapi | None = None
_corpus_metadata: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9][a-z0-9\-]*\b', text)
    return tokens


def build_index(chunks: list[dict]) -> None:
    """
    Build BM25 index from a list of chunk dicts.
    Persists to disk for future loads.
    """
    global _bm25, _corpus_metadata
    corpus = [_tokenize(c["text"]) for c in chunks]
    _bm25 = BM25Okapi(corpus)
    _corpus_metadata = chunks

    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": _bm25, "metadata": _corpus_metadata}, f)


def load_index() -> bool:
    """Load BM25 index from disk. Returns True if successful."""
    global _bm25, _corpus_metadata
    if _bm25 is not None:
        return True
    if not Path(BM25_INDEX_PATH).exists():
        return False
    with open(BM25_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    _bm25 = data["bm25"]
    _corpus_metadata = data["metadata"]
    return True


def search(query: str, top_k: int = 50) -> list[dict[str, Any]]:
    """
    BM25 keyword search.
    Returns list of chunk dicts with 'bm25_score' field.
    """
    if not load_index():
        return []

    query_tokens = _tokenize(query)
    scores = _bm25.get_scores(query_tokens)

    # Get top_k indices
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:  # Skip zero-score results
            results.append({
                "bm25_score": float(scores[idx]),
                **_corpus_metadata[idx],
            })
    return results


def get_corpus_size() -> int:
    """Return number of documents in BM25 index."""
    if not load_index():
        return 0
    return len(_corpus_metadata)
