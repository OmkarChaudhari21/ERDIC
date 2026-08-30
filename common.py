"""Shared UI chrome: the backend connection sidebar every page uses.

One place builds the client, so every page talks to the same backend with the same key
and shows the same connection status. The client is cached per (URL, key) pair; changing
either in the sidebar builds a fresh one.
"""

from __future__ import annotations

import os

import streamlit as st

from erdic_ui.client import BackendClient, BackendError


@st.cache_resource(show_spinner=False)
def get_client(base_url: str, api_key: str) -> BackendClient:
    return BackendClient(base_url, api_key)


def connection_sidebar() -> BackendClient:
    """Render the connection controls and return the configured client."""
    with st.sidebar:
        st.title("ERDIC")
        st.caption("Enterprise Research & Decision Intelligence Copilot")
        base_url = st.text_input(
            "Backend URL",
            value=os.environ.get("ERDIC_UI_BACKEND_URL", "http://127.0.0.1:8000"),
            key="backend_url",
        )
        api_key = st.text_input(
            "API key",
            value=os.environ.get("ERDIC_UI_API_KEY", ""),
            type="password",
            key="backend_api_key",
        )
        client = get_client(base_url, api_key)
        try:
            ready = client.ready()
            st.success(f"backend ready · pgvector {ready['checks'].get('pgvector', '?')}")
        except BackendError as exc:
            st.error(f"backend unavailable: {exc.message}")
        return client
