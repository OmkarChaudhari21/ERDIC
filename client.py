"""A thin HTTP client for the ERDIC backend.

Deliberately dumb: it sends requests, translates the backend's error envelope into one
exception type, and hands back the response fields the UI renders. No logic that exists
in the backend is repeated here -- if the UI needs a number, the API grows it.

The transport is injectable so tests can mount the real FastAPI application in-process
(``httpx.ASGITransport``) and exercise the exact client the UI ships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

DEFAULT_TIMEOUT_SECONDS = 300.0


def normalise_api_key(value: str) -> str:
    """One usable key from however the value was plumbed here.

    The backend's ERDIC_API_KEYS accepts a JSON array or a comma-separated list, and
    deployment glue (compose interpolation, copied `.env` lines) hands the UI that same
    raw string. Mirroring the backend's parsing -- including its whitespace stripping --
    and taking the first key means a value that authenticates the backend also
    authenticates against it, instead of failing on an invisible bracket or space.
    """
    text = value.strip()
    if text.startswith("["):
        import json

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(items, list) and items:
            return str(items[0]).strip()
        return ""
    return text.split(",", 1)[0].strip()


class BackendError(Exception):
    """The backend refused or failed; carries what its error envelope said."""

    def __init__(self, message: str, *, status: int | None = None,
                 code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code


@dataclass(frozen=True, slots=True)
class Citation:
    """One citation exactly as the backend reported it."""

    index: int
    label: str
    quote: str
    document_title: str


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """A document as the backend lists it."""

    id: str
    title: str
    status: str
    media_type: str | None
    byte_size: int | None
    page_count: int | None
    created_at: str


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One retrieved chunk with every stage's score, as the backend reported them."""

    rank: int
    document_title: str
    page_number: int | None
    section: str | None
    content: str
    score: float
    #: Per-stage scores keyed ``dense``, ``bm25``, ``rrf``, ``rerank``; a stage that did
    #: not touch this hit is simply absent.
    scores: dict[str, float]
    #: 1-based rank each retriever gave this hit before fusion.
    ranks: dict[str, int]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A ``/search`` response: the hits plus how they were produced."""

    hits: list[SearchHit]
    retriever: str
    searchable_chunks: int
    reranker: str | None
    stages: dict[str, Any] | None
    duration_ms: float


@dataclass(frozen=True, slots=True)
class Answer:
    """The ``/query`` response fields the chat page renders."""

    question: str
    final_answer: str
    citations: list[Citation]
    confidence: float
    caveats: list[str]
    retry_count: int
    evidence_status: str
    verification_verdict: str | None
    query_type: str
    router: str | None
    subqueries: list[str]
    retrieved: int
    evidence: int
    duration_ms: float
    tool_results: list[dict[str, Any]] = field(default_factory=list)


class BackendClient:
    """Talks to the FastAPI backend and nothing else."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        api_key = normalise_api_key(api_key)
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key} if api_key else {},
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        return self._get("/api/v1/health")

    def ready(self) -> dict[str, Any]:
        return self._get("/api/v1/ready")

    def ask(self, question: str) -> Answer:
        """Run one question through the backend's full agent workflow."""
        payload = self._post("/api/v1/query", {"question": question})
        workflow = payload.get("workflow", {})
        return Answer(
            question=payload.get("question", question),
            final_answer=payload.get("final_answer", ""),
            citations=[
                Citation(
                    index=int(citation.get("index", 0)),
                    label=str(citation.get("label", "")),
                    quote=str(citation.get("quote", "")),
                    document_title=str(citation.get("document_title", "")),
                )
                for citation in payload.get("citations", [])
            ],
            confidence=float(payload.get("confidence", 0.0)),
            caveats=[str(caveat) for caveat in payload.get("caveats", [])],
            retry_count=int(workflow.get("retry_count", 0)),
            evidence_status=str(workflow.get("evidence_status", "unknown")),
            verification_verdict=workflow.get("verification_verdict"),
            query_type=str(workflow.get("query_type", "")),
            router=workflow.get("router"),
            subqueries=[str(item) for item in workflow.get("subqueries", [])],
            retrieved=int(workflow.get("retrieved", 0)),
            evidence=int(workflow.get("evidence", 0)),
            duration_ms=float(payload.get("duration_ms", 0.0)),
            tool_results=list(payload.get("tool_results", [])),
        )

    # ---- documents ----

    def list_documents(self, *, limit: int = 100) -> tuple[list[DocumentInfo], int]:
        payload = self._get(f"/api/v1/documents?limit={limit}")
        return (
            [_document_info(item) for item in payload.get("items", [])],
            int(payload.get("total", 0)),
        )

    def upload_document(
        self, filename: str, data: bytes, media_type: str
    ) -> tuple[DocumentInfo, bool]:
        """Upload one file; returns the document and whether it was a duplicate."""
        payload = self._handle(
            lambda: self._client.post(
                "/api/v1/documents/upload",
                files={"file": (filename, data, media_type)},
            )
        )
        return _document_info(payload["document"]), bool(payload.get("duplicate"))

    def ingest_document(self, document_id: str, *, replace: bool = False) -> dict[str, Any]:
        """Run the backend's parse-chunk-embed chain for one document."""
        suffix = "?replace=true" if replace else ""
        return self._post(f"/api/v1/documents/{document_id}/ingest{suffix}", {})

    def delete_document(self, document_id: str) -> None:
        self._handle(lambda: self._client.delete(f"/api/v1/documents/{document_id}"))

    # ---- retrieval ----

    def search(
        self, query: str, *, retriever: str = "hybrid", top_k: int | None = None
    ) -> SearchResult:
        payload: dict[str, Any] = {"query": query, "retriever": retriever}
        if top_k is not None:
            payload["top_k"] = top_k
        body = self._post("/api/v1/search", payload)
        return SearchResult(
            hits=[
                SearchHit(
                    rank=position,
                    document_title=str(hit.get("document_title", "")),
                    page_number=hit.get("page_number"),
                    section=hit.get("section"),
                    content=str(hit.get("content", "")),
                    score=float(hit.get("score", 0.0)),
                    scores={k: float(v) for k, v in (hit.get("scores") or {}).items()},
                    ranks={k: int(v) for k, v in (hit.get("ranks") or {}).items()},
                )
                for position, hit in enumerate(body.get("hits", []), start=1)
            ],
            retriever=str(body.get("retriever", retriever)),
            searchable_chunks=int(body.get("searchable_chunks", 0)),
            reranker=body.get("reranker"),
            stages=body.get("stages"),
            duration_ms=float(body.get("duration_ms", 0.0)),
        )

    # ---- transport plumbing ----

    def _get(self, path: str) -> dict[str, Any]:
        return self._handle(lambda: self._client.get(path))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._handle(lambda: self._client.post(path, json=payload))

    def _handle(self, request: Any) -> dict[str, Any]:
        try:
            response = request()
        except httpx.HTTPError as exc:
            raise BackendError(f"could not reach the backend: {exc}") from exc
        try:
            body = response.json()
        except ValueError:
            body = {}
        if response.status_code >= 400:
            envelope = body.get("error", {}) if isinstance(body, dict) else {}
            raise BackendError(
                str(envelope.get("message") or f"backend returned {response.status_code}"),
                status=response.status_code,
                code=envelope.get("code"),
            )
        if not isinstance(body, dict):
            raise BackendError("the backend returned an unexpected response shape")
        return body


def _document_info(item: dict[str, Any]) -> DocumentInfo:
    return DocumentInfo(
        id=str(item.get("id", "")),
        title=str(item.get("title", "")),
        status=str(item.get("status", "")),
        media_type=item.get("media_type"),
        byte_size=item.get("byte_size"),
        page_count=item.get("page_count"),
        created_at=str(item.get("created_at", "")),
    )
