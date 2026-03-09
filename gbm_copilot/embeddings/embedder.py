"""
BGE-M3 Embedder
Medical-domain text embedder using BAAI/bge-m3 via fastembed.
Fallback to OpenAI text-embedding-3-small if fastembed unavailable.
"""
from __future__ import annotations

import numpy as np
from typing import Any

from gbm_copilot.config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

_local_embedder = None
_openai_client = None
EMBEDDING_DIM = 1024  # BGE-M3 output dimension


def _get_local_embedder():
    global _local_embedder
    if _local_embedder is None:
        try:
            from fastembed import TextEmbedding
            _local_embedder = TextEmbedding(model_name=EMBEDDING_MODEL)
        except Exception:
            _local_embedder = False  # Mark as unavailable
    return _local_embedder if _local_embedder is not False else None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts.
    Uses BGE-M3 (local) if available, else OpenAI embeddings.
    Returns numpy array of shape (n_texts, embedding_dim).
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM))

    local = _get_local_embedder()
    if local is not None:
        embeddings = list(local.embed(texts))
        arr = np.array(embeddings, dtype=np.float32)
        return arr

    # Fallback: OpenAI
    client = _get_openai_client()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    arr = np.array([e.embedding for e in response.data], dtype=np.float32)
    return arr


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns shape (embedding_dim,)."""
    result = embed_texts([query])
    return result[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vector a and matrix b.
    a: shape (dim,), b: shape (n, dim)
    Returns shape (n,)
    """
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    b_normalized = b / b_norms
    return b_normalized @ a_norm


def get_embedding_dim() -> int:
    """Return the embedding dimension for the configured model."""
    local = _get_local_embedder()
    if local is not None:
        # BGE-M3 dim
        return 1024
    else:
        # OpenAI text-embedding-3-small dim
        return 1536
