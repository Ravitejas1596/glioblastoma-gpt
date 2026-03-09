"""
Centralized LLM Client Factory for GlioblastomaGPT
====================================================
Supports two providers, configured via LLM_PROVIDER env variable:

  LLM_PROVIDER=groq    → Groq (free tier, llama-3.3-70b-versatile)
  LLM_PROVIDER=openai  → OpenAI GPT-4o (paid)

Both return an AsyncOpenAI-compatible client — zero changes needed
in agent code, because Groq's Python SDK is 100% OpenAI-compatible.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root (works both locally and on Streamlit Cloud)
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# Try loading from Streamlit secrets (for Streamlit Cloud deployment)
try:
    import streamlit as st
    _secrets = st.secrets
except Exception:
    _secrets = {}


def _get_secret(key: str, default: str = "") -> str:
    """Get a secret from env, .env, or Streamlit secrets (in that order)."""
    val = os.environ.get(key, "")
    if val:
        return val
    if _secrets and hasattr(_secrets, "get"):
        return _secrets.get(key, default)
    return default


# ── Provider selection ────────────────────────────────────────────────────────
LLM_PROVIDER: str = _get_secret("LLM_PROVIDER", "groq").lower()

# ── Groq settings ─────────────────────────────────────────────────────────────
GROQ_API_KEY: str = _get_secret("GROQ_API_KEY", "")
GROQ_MODEL: str = _get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── OpenAI settings (fallback) ────────────────────────────────────────────────
OPENAI_API_KEY: str = _get_secret("OPENAI_API_KEY", "")
OPENAI_MODEL: str = _get_secret("OPENAI_MODEL", "gpt-4o")


def get_client():
    """
    Returns an async LLM client.
    - Groq client (when LLM_PROVIDER=groq or GROQ_API_KEY is set)
    - OpenAI client (when LLM_PROVIDER=openai)

    Both clients expose the same .chat.completions.create(...) interface.
    """
    if LLM_PROVIDER == "groq" or (GROQ_API_KEY and not OPENAI_API_KEY):
        from groq import AsyncGroq
        return AsyncGroq(api_key=GROQ_API_KEY)
    else:
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=OPENAI_API_KEY)


def get_model() -> str:
    """Returns the model name for the active provider."""
    if LLM_PROVIDER == "groq" or (GROQ_API_KEY and not OPENAI_API_KEY):
        return GROQ_MODEL
    return OPENAI_MODEL
