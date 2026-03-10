"""
Centralized LLM Client Factory for GlioblastomaGPT
====================================================
Reads API keys dynamically (never cached at import time) so Streamlit Cloud
secrets are always picked up, even after late injection.

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
    """
    Read a config value. Tries in order:
      1. os.environ  (works locally and after streamlit_app.py injection)
      2. st.secrets  (Streamlit Cloud — dict-style access, not .get())
      3. default
    """
    # 1. Environment variable (fastest path)
    val = os.environ.get(key, "")
    if val:
        return val

    # 2. Streamlit secrets (Streamlit Cloud)
    try:
        import streamlit as st
        # Use dict-style access so KeyError is raised if missing (not silent "")
        val = str(st.secrets[key])
        if val:
            # Persist into os.environ so all threads/calls see it going forward
            os.environ[key] = val
            return val
    except KeyError:
        pass  # Key genuinely not in secrets
    except Exception:
        pass  # st not available (local dev without streamlit)

    return default


def get_client():
    """
    Return a ready-to-use async LLM client.
    Keys are read fresh on every call — safe for Streamlit Cloud.
    """
    provider = _get("LLM_PROVIDER", "groq").lower()
    groq_key = _get("GROQ_API_KEY", "")
    openai_key = _get("OPENAI_API_KEY", "")

    if provider == "groq" or (groq_key and not openai_key):
        if not groq_key:
            raise ValueError(
                "GROQ_API_KEY not found. On Streamlit Cloud go to: "
                "App Settings → Secrets and add:\n"
                'GROQ_API_KEY = "gsk_your_key_here"'
            )
        from groq import AsyncGroq
        return AsyncGroq(api_key=groq_key)
    else:
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found.")
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
