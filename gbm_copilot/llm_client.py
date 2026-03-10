"""
Centralized LLM Client Factory for GlioblastomaGPT
====================================================
Reads API keys DYNAMICALLY on every call (not at import time) so that
Streamlit Cloud secrets injected into os.environ are always picked up.

  LLM_PROVIDER=groq    → Groq free tier (llama-3.3-70b-versatile)
  LLM_PROVIDER=openai  → OpenAI GPT-4o (paid)
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env for local development
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")


def _get(key: str, default: str = "") -> str:
    """Read a config value from os.environ first, then st.secrets (Streamlit Cloud)."""
    val = os.environ.get(key, "")
    if val:
        return val
    # Streamlit Cloud secrets fallback (read every call so late-injection works)
    try:
        import streamlit as st
        val = st.secrets.get(key, default)
        if val:
            # Also inject into os.environ so subprocesses / threads see it
            os.environ[key] = str(val)
        return str(val)
    except Exception:
        return default


def get_client():
    """
    Return a ready-to-use async LLM client.
    Keys are read fresh from env/secrets on every call — safe for Streamlit Cloud.
    """
    provider = _get("LLM_PROVIDER", "groq").lower()
    groq_key = _get("GROQ_API_KEY", "")
    openai_key = _get("OPENAI_API_KEY", "")

    if provider == "groq" or (groq_key and not openai_key):
        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Add it to Streamlit Cloud Secrets: "
                'GROQ_API_KEY = "gsk_..."'
            )
        from groq import AsyncGroq
        return AsyncGroq(api_key=groq_key)
    else:
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        from openai import AsyncOpenAI
        return AsyncOpenAI(api_key=openai_key)


def get_model() -> str:
    """Return the model name for the active provider."""
    provider = _get("LLM_PROVIDER", "groq").lower()
    groq_key = _get("GROQ_API_KEY", "")
    openai_key = _get("OPENAI_API_KEY", "")

    if provider == "groq" or (groq_key and not openai_key):
        return _get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return _get("OPENAI_MODEL", "gpt-4o")
