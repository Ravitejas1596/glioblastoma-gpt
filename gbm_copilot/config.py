"""
Global configuration for GlioblastomaGPT.
All settings loaded from environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_EMBEDDING_MODEL: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")

# ── Retrieval ─────────────────────────────────────────────────────────────────
RetrievalMode = Literal["pinecone", "numpy"]
RETRIEVAL_MODE: RetrievalMode = os.environ.get("RETRIEVAL_MODE", "numpy")  # type: ignore

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX_NAME", "gbm-copilot")
PINECONE_ENVIRONMENT: str = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1-aws")

# ── LangSmith ─────────────────────────────────────────────────────────────────
LANGCHAIN_TRACING_V2: bool = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_API_KEY: str = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT: str = os.environ.get("LANGCHAIN_PROJECT", "gbm-copilot")

# ── PubMed ────────────────────────────────────────────────────────────────────
NCBI_EMAIL: str = os.environ.get("NCBI_EMAIL", "gbm-copilot@example.com")

# ── Ingest ────────────────────────────────────────────────────────────────────
INGEST_QUICK_MODE: bool = os.environ.get("INGEST_QUICK_MODE", "false").lower() == "true"
CHUNK_SIZE_TOKENS: int = int(os.environ.get("CHUNK_SIZE_TOKENS", "256"))
CHUNK_OVERLAP_PCT: float = float(os.environ.get("CHUNK_OVERLAP_PCT", "0.20"))
MAX_PUBMED_RESULTS: int = int(os.environ.get("MAX_PUBMED_RESULTS", "50000"))
QUICK_PUBMED_RESULTS: int = 500

# ── Safety ────────────────────────────────────────────────────────────────────
MIN_CONFIDENCE_THRESHOLD: float = float(os.environ.get("MIN_CONFIDENCE_THRESHOLD", "0.50"))
FAITHFULNESS_WARNING_THRESHOLD: float = float(os.environ.get("FAITHFULNESS_WARNING_THRESHOLD", "0.70"))

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.environ.get("API_HOST", "0.0.0.0")
API_PORT: int = int(os.environ.get("API_PORT", "8000"))
CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:3000"]

# ── MCP ───────────────────────────────────────────────────────────────────────
MCP_PORT: int = int(os.environ.get("MCP_PORT", "8001"))

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR: Path = ROOT_DIR / "data"
NUMPY_INDEX_PATH: Path = DATA_DIR / "numpy_index"
BM25_INDEX_PATH: Path = DATA_DIR / "bm25_index.pkl"
CHUNKS_PATH: Path = DATA_DIR / "chunks.jsonl"

DATA_DIR.mkdir(exist_ok=True)
NUMPY_INDEX_PATH.mkdir(exist_ok=True)

# ── Ontology ──────────────────────────────────────────────────────────────────
ONTOLOGY_PATH: Path = ROOT_DIR / "gbm_copilot" / "ontology" / "gbm_ontology.json"

# ── Literacy modes ────────────────────────────────────────────────────────────
LiteracyMode = Literal["patient", "caregiver", "clinician"]
LITERACY_MODE_DESCRIPTIONS: dict[str, str] = {
    "patient": "plain English, avoid jargon, compassionate tone",
    "caregiver": "clear but more detailed, some medical terms with explanations",
    "clinician": "technical, full medical terminology, citations formatted as references",
}
