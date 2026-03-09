"""
Root-level Streamlit entry point for Streamlit Community Cloud.
This calls the pipeline directly (no FastAPI hop needed).
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load secrets from Streamlit (when running on Streamlit Cloud)
# This must happen before importing gbm_copilot modules
try:
    import streamlit as st

    # Inject Streamlit secrets into environment variables so config.py picks them up
    for k, v in st.secrets.items():
        if isinstance(v, str) and k not in os.environ:
            os.environ[k] = v
except Exception:
    pass  # Running locally with .env file — no problem

# Now import and run the actual app
from gbm_copilot.ui.app_standalone import main

main()
