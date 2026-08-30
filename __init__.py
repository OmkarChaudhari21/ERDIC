"""The Streamlit research UI.

    client.py  a thin, typed HTTP client for the FastAPI backend
    app.py     (one level up) the Research Chat page

The UI holds no retrieval, generation, or validation logic: every answer, citation,
confidence figure, and workflow fact on screen came from the backend's ``/query``
response, rendered as received. Duplicating backend behaviour in a frontend is how the
two silently disagree.
"""
