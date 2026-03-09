"""
Embedder — Multi-backend text embedder for GlioblastomaGPT.

Priority order (fastest install wins on cloud):
  1. sentence-transformers/all-MiniLM-L6-v2  (lightweight, 80MB, fast)
  2. fastembed BAAI/bge-m3                   (best quality, 600MB — local dev)
  3. OpenAI text-embedding-3-small           (fallback if no local model)

On Streamlit Cloud, sentence-transformers is used (installed in requirements.txt).
On local dev with fastembed installed, BGE-M3 is used automatically.
"""
from __future__ import annotations

import numpy as np

from gbm_copilot.config import EMBEDDING_MODEL, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

_local_embedder = None
_openai_client = None

# sentence-transformers model (fast, light — used on Streamlit Cloud)
_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_ST_DIM = 384

# fastembed BGE-M3 (best quality — local dev)
_BGE_DIM = 1024

EMBEDDING_DIM = _ST_DIM  # Set at module level; updated once embedder is loaded


def _get_local_embedder():
    """Try fastembed first (best quality), then sentence-transformers (cloud-friendly)."""
    global _local_embedder, EMBEDDING_DIM

    if _local_embedder is not None:
        return _local_embedder if _local_embedder is not False else None

    # 1. Try fastembed (BGE-M3) — only available if locally installed
    try:
        from fastembed import TextEmbedding
        _local_embedder = ("fastembed", TextEmbedding(model_name=EMBEDDING_MODEL))
        EMBEDDING_DIM = _BGE_DIM
        return _local_embedder
    except Exception:
        pass

    # 2. Try sentence-transformers (installed on Streamlit Cloud)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_ST_MODEL)
        _local_embedder = ("st", model)
        EMBEDDING_DIM = _ST_DIM
        return _local_embedder
    except Exception:
        pass

    _local_embedder = False
    return None


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts.
    Returns numpy array of shape (n_texts, embedding_dim).
    """
    if not texts:
        return np.zeros((0, EMBEDDING_DIM))

    local = _get_local_embedder()
    if local is not None:
        kind, model = local
        if kind == "fastembed":
            embeddings = list(model.embed(texts))
            return np.array(embeddings, dtype=np.float32)
        elif kind == "st":
            embeddings = model.encode(texts, normalize_embeddings=True)
            return np.array(embeddings, dtype=np.float32)

    # Fallback: OpenAI embeddings
    client = _get_openai_client()
    response = client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([e.embedding for e in response.data], dtype=np.float32)


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


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
    """Return the embedding dimension for the active model."""
    _get_local_embedder()  # Ensure model is loaded so EMBEDDING_DIM is set
    return EMBEDDING_DIM
