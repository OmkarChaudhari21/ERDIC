"""The ERDIC research UI: Research Chat (main page).

A thin window onto the FastAPI backend -- every fact on screen was reported by the API.
Further pages live under ``pages/``. Run with:

    .venv/bin/streamlit run frontend/app.py

Configuration comes from the sidebar, defaulting to ERDIC_UI_BACKEND_URL
(http://127.0.0.1:8000) and ERDIC_UI_API_KEY.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from erdic_ui.common import connection_sidebar
from erdic_ui.views import chat

st.set_page_config(page_title="ERDIC Research", page_icon="📚", layout="wide")

chat.render(connection_sidebar())
