# Enterprise Research & Decision Intelligence Copilot

Ingests enterprise documents, runs multi-step research over them, and produces grounded
answers with validated inline citations and a full audit trail.

The complete pipeline is implemented and tested end to end: document ingestion (PDF,
text, Markdown) with token-aware chunking; hybrid retrieval (dense pgvector + genuine
BM25 + reciprocal rank fusion + optional cross-encoder reranking); a LangGraph research
workflow with query classification, decomposition, evidence verification, citation
validation, tools, and a bounded retry loop; a provider-agnostic LLM layer (offline by
default); a LoRA-fine-tuned query router; a native RAG evaluation framework with a
56-question golden dataset; a Streamlit frontend; structured logging, Prometheus
metrics, and optional MLflow tracking; and a Docker compose stack for all of it.

Further reading:

| Document | Covers |
|----------|--------|
| [docs/architecture.md](docs/architecture.md) | system shape, the RAG pipeline, the LangGraph workflow, decisions, limitations, future work |
| [docs/api.md](docs/api.md) | every endpoint, auth, the error contract |
| [docs/evaluation.md](docs/evaluation.md) | metrics, the golden dataset, baseline vs improved |
| [docs/fine-tuning.md](docs/fine-tuning.md) | the query router: dataset, training, measured results |
| [docs/development.md](docs/development.md) | setup, testing, migrations, Docker, conventions |

## Requirements

- Python 3.11 or 3.12 (3.11 is what this is verified against; 3.14 is excluded because
  parts of the ML stack lack wheels for it)
- PostgreSQL 17 with the pgvector extension
- Everything else is optional extras: `embeddings` (real local embedding models and the
  cross-encoder reranker), `llm` (local transformers inference), `training` (router
  fine-tuning + MLflow), `frontend` (Streamlit), `tiktoken` (exact OpenAI token counts)

## Setup

```bash
make dev        # venv on Python 3.11, editable install, .env from .env.example
make db-setup   # role, databases, and pgvector extension (needs a superuser)
make migrate    # apply migrations
make dbcheck    # confirm connection, pgvector, and migration state
make check      # lint + typecheck + tests
make serve      # http://127.0.0.1:8000  (docs at /docs in local env)
```

On macOS, `brew install postgresql@17 pgvector` then `make db-start` provides the server.

### Docker

The whole stack runs as containers, with no local Python or PostgreSQL:

```bash
docker compose up --build
# API      http://127.0.0.1:8000
# UI       http://127.0.0.1:8501
# MLflow   http://127.0.0.1:5001
# Postgres 127.0.0.1:5433
```

The `api` container waits for the database, applies migrations, then serves; the
`frontend` (Streamlit) starts once the API and MLflow are healthy and reaches them by
service name. Every container carries a health check, and `docker compose ps` shows
them. Configuration comes from the shell environment or the git-ignored `.env` (compose
interpolates `${VAR:-default}`): no secrets exist in the Dockerfiles, the compose file,
or the images. The defaults run the offline `fake` providers; set `ERDIC_LLM_*` for a
real model, and build the API with `--build-arg EXTRAS=embeddings,llm` for real local
models. Databases, uploads, and MLflow runs persist in named volumes across restarts;
locally produced `artifacts/` mount read-only into the UI's dashboards.

`make docker-up`, `make docker-logs`, and `make docker-down` wrap the common commands.

## Layout

```
src/erdic/
  main.py          FastAPI application factory and ASGI entrypoint
  cli.py           `erdic serve | config | initdb | dbcheck`
  core/            configuration, logging (with secret redaction), errors, metrics
  api/             routers, dependencies, middleware, HTTP error translation
  db/              declarative base, engine/session handling, pgvector, ORM models
  repositories/    all query construction against the models
  ingestion/       upload, parsing, cleaning, chunking, embedding, BM25 indexing
  embeddings/      embedding provider abstraction (fake, sentence-transformers)
  retrieval/       dense, BM25, RRF fusion, cross-encoder reranking, audit persistence
  llm/             LLM provider abstraction (fake, openai-compatible, ollama, huggingface)
  generation/      versioned prompts, structured-output schemas, the prompt runner
  synthesis.py     grounded answer generation with citation validation
  tools/           calculator (safe arithmetic) and metadata lookup
  agents/          the LangGraph research workflow: state, twelve nodes, retry routing
  routing/         query routers: heuristic baseline and the fine-tuned adapter
  evaluation/      golden dataset, metrics, harness, baseline-vs-improved runner
  training/        routing dataset generation, LoRA/QLoRA training, MLflow tracking
  schemas/         Pydantic API contract
migrations/        Alembic environment and revisions
frontend/          Streamlit UI: chat, documents, retrieval inspector, dashboards
evaluation/        repo-root shim so `python -m evaluation.run` works from a checkout
tests/unit/        fast isolated tests
tests/integration/ tests across components, against real PostgreSQL
scripts/           training/evaluation entry points and live-stack verification
docs/              architecture, API, evaluation, fine-tuning, development
```

## Database

PostgreSQL with pgvector is the only supported backend: embeddings live in `vector`
columns, which nothing else can represent, so a portable-looking fallback would mislead.
`ERDIC_DATABASE_URL` accepts `postgres://` and bare `postgresql://` forms and rewrites
them onto the psycopg3 driver, so a connection string copied from a hosting provider
works untouched.

Extension installation and schema migration are deliberately separate concerns:

| Step             | Who runs it   | Why                                                     |
|------------------|---------------|---------------------------------------------------------|
| `make db-setup`  | a superuser   | pgvector is not a *trusted* extension, so `CREATE EXTENSION` needs superuser |
| `make migrate`   | the app role  | schema changes only; asserts the extension is present   |

The first migration therefore no-ops when an administrator has already installed the
extension, installs it when the migrating role is privileged, and otherwise fails with
instructions. Its downgrade is intentionally a no-op, because the extension is shared
infrastructure the app role does not own.

Migrations are the mechanism of record. `make migration m="..."` autogenerates one, and a
test fails if models and migrations ever drift apart.

## Schema

```
Document 1──* DocumentChunk
Query    1──* RetrievalResult *──1 DocumentChunk
Query    1──* Answer          1──* Evaluation
Query    1──* Evaluation
```

| Table               | Holds                                                        |
|---------------------|--------------------------------------------------------------|
| `documents`         | one ingested file, its provenance and status                 |
| `document_chunks`   | retrievable spans plus their `vector(1024)` embedding        |
| `queries`           | a research question, its filters and sub-questions           |
| `retrieval_results` | which chunk was returned at which rank, and how it scored    |
| `chunk_terms`       | the inverted index BM25 scores from: term frequency per chunk |
| `answers`           | a generated brief, its inline citations and token cost       |
| `evaluations`       | one scored metric against an answer or a retrieval run       |

Every table has a UUID primary key, timezone-aware `created_at`/`updated_at`, and a JSONB
`metadata` column (exposed as `.meta`, since `metadata` is reserved by SQLAlchemy).

Delete behaviour is deliberately not uniform. Chunks cascade from their document, because
a chunk without its source cannot be cited. `retrieval_results.chunk_id` is instead
`ON DELETE SET NULL`: those rows are audit evidence, so the record that something was
retrieved at a given rank and score has to outlive the document, and `document_title` is
denormalised onto the row so it stays readable.

Embedding width is a code constant (`EMBEDDING_DIMENSIONS`, 1024), not a setting: a
pgvector column has a fixed dimension and its ANN index is built for that width, so
changing it requires a migration *and* re-embedding every chunk. An environment variable
would let those drift apart silently. Retrieval indexes are in place already: HNSW with
`vector_cosine_ops` for the vector half, a GIN index over `to_tsvector('english', content)`
for the keyword half.

## Configuration

All configuration is typed in `erdic.core.config.Settings` and read from `ERDIC_*`
environment variables or `.env`. See `.env.example` for every knob. Nothing else in the
codebase reads `os.environ`.

Two rails are enforced at startup: outside `ERDIC_ENV=local`, the app refuses to run
with no API keys, with debug enabled, or with wildcard CORS. API docs are served only in
the local environment.

## Endpoints

| Method | Path                        | Auth      | Purpose                        |
|--------|-----------------------------|-----------|--------------------------------|
| GET    | `/api/v1/health`            | public    | liveness, touches no dependency |
| GET    | `/api/v1/ready`             | public    | readiness: database reachable *and* pgvector installed |
| POST   | `/api/v1/documents/upload`  | `X-API-Key` | ingest a PDF, text, or Markdown file |
| GET    | `/api/v1/documents`         | `X-API-Key` | list documents, optionally by status |
| GET    | `/api/v1/documents/{id}`    | `X-API-Key` | one document                   |
| POST   | `/api/v1/documents/{id}/parse` | `X-API-Key` | parse and chunk, `?replace=true` to re-chunk |
| POST   | `/api/v1/documents/{id}/embed` | `X-API-Key` | embed chunks, `?replace=true` to re-embed |
| POST   | `/api/v1/documents/{id}/ingest` | `X-API-Key` | the whole chain: parse, chunk, embed |
| GET    | `/api/v1/documents/{id}/chunks` | `X-API-Key` | chunks in reading order   |
| DELETE | `/api/v1/documents/{id}`    | `X-API-Key` | delete a document and its chunks |
| POST   | `/api/v1/search`            | `X-API-Key` | hybrid, dense, or BM25 search; `persist` records the run |
| POST   | `/api/v1/ask`               | `X-API-Key` | single-pass grounded answer: retrieve, generate, validate citations |
| POST   | `/api/v1/query`             | `X-API-Key` | the full LangGraph research workflow |
| GET    | `/api/v1/metrics`           | public    | Prometheus text exposition of every application metric |

Full request/response shapes are in [docs/api.md](docs/api.md). Content-bearing routers
declare `require_api_key` at router level rather than per route, so a route added later
cannot be left public by omission; `require_api_key` fails closed when no keys are
configured, and a test walks the OpenAPI schema calling every route unauthenticated to
pin the auth surface.

## Uploads

PDF, plain text, and Markdown. The supported set is a code constant, not configuration:
accepting a format means a parser exists for it, so an environment variable listing extra
formats would promise something nothing keeps.

A single pass over the request body does size accounting, SHA-256 hashing, format
validation, and the write. Nothing is buffered whole, and an oversized body is abandoned as
soon as it crosses the limit instead of after it has all arrived.

Security decisions worth knowing:

- **Content decides the format.** The declared `Content-Type` and the filename extension
  are both attacker-controlled. A file presented as PDF whose bytes lack `%PDF-` is
  rejected, rather than reinterpreted as text and stored under a trusted name.
- **Text is verified as UTF-8 incrementally**, so a multi-byte character split across
  chunks does not read as corrupt. NUL bytes are refused: they are valid UTF-8 but signal
  binary content and truncate strings in C-backed consumers.
- **Storage is keyed by a generated UUID**, sharded two levels deep
  (`ab/cd/<uuid>.<ext>`). No part of the path comes from client input, so traversal has
  nothing to aim at. The sanitised filename is kept as metadata only.
- **Writes are atomic.** Bytes go to a temporary file and are then `os.replace`d, so a
  reader never sees a partial document. Files are created `0600`.
- **The file is written before the row is inserted**, and removed if the insert fails. The
  reverse order risks a document whose bytes are missing, which breaks every later read; an
  orphaned file is merely wasted space.
- **Duplicates are content-addressed.** An identical SHA-256 returns `200` with the
  existing document and `duplicate: true`, instead of `201`. A concurrent upload of the
  same bytes is resolved against the unique index, so a lost race still returns the winner
  rather than an error.

An uploaded document is `PENDING`, not `READY`: the bytes are held but nothing has been
parsed, chunked, or embedded, and a document claiming to be ready would be retrieved.

## Parsing and chunking

`POST /api/v1/documents/{id}/parse` extracts text, cleans it, and persists chunks, moving
the document to `CHUNKED`. It runs synchronously — there is no worker yet, so a large PDF
holds the request open. `?replace=true` re-chunks a document that already has chunks, which
is what makes a chunk-size change applicable to existing documents; without it a second run
is refused, because silently duplicating chunks would double every retrieval hit.

What each format can honestly provide differs, and the parsers do not pretend otherwise:

| Format   | Page numbers | Sections                    |
|----------|--------------|-----------------------------|
| PDF      | yes, exact   | no — see below              |
| Markdown | n/a          | yes, from heading syntax    |
| Text     | n/a          | no — see below              |

PDF sections are not inferred because text extraction discards the font size and weight that
would identify a heading; plain-text sections are not inferred because a short capitalised
line is as likely to be an address or a table caption. A wrong section label on a citation
misleads a reader more than a missing one. PDF *document* metadata (title, author, dates) is
captured under a `pdf` key in the document's metadata, kept separate because an embedded
title is frequently a template leftover.

Cleaning repairs layout damage without changing wording: words hyphenated across line breaks
are rejoined, hard-wrapped lines are unwrapped back into paragraphs, ligatures and
full-width forms are folded via NFKC so `ﬁle` matches `file`, and zero-width characters are
removed. Markdown is cleaned without unwrapping, since a line break there can be structural
and code fences are passed through untouched.

Chunking is token-aware, defaulting to **800-token chunks with 120-token overlap**
(`ERDIC_CHUNK_SIZE_TOKENS`, `ERDIC_CHUNK_OVERLAP_TOKENS`). Chunks are packed to the budget at
sentence boundaries rather than cut at an exact token count, because a chunk ending
mid-sentence retrieves badly and quotes worse. Overlap is carried as whole trailing
sentences, so an overlap budget smaller than one sentence yields no overlap. A single
sentence that exceeds the budget on its own — tables and reference lists extract this way —
is hard-split on word boundaries rather than emitted over budget, which would fail later at
the embedding call.

A chunk never spans two pages or two sections. Provenance has to stay exact: a chunk
labelled page 1 while containing page 2 text produces a citation pointing where the text is
not.

Token counts come from a tokenizer chosen by `ERDIC_TOKENIZER`. The default `heuristic` is
deterministic and works offline. `tiktoken` gives exact counts for OpenAI models but
downloads its encoding tables on first use, so it is opt-in and installed via the
`tiktoken` extra. Whichever produced a chunk set is recorded on the document alongside the
chunk size and overlap, since a stored chunk set is only interpretable next to the budget
that produced it.

A PDF with no text layer fails with an explicit error rather than succeeding with zero
chunks — it needs OCR, which is not implemented, and a document that reached `READY` empty
would be silently unretrievable.

## Embeddings

`POST /api/v1/documents/{id}/embed` embeds a document's chunks into the `vector(1024)`
column; `POST /api/v1/documents/{id}/ingest` runs the whole chain — parse, clean, chunk,
embed — in one transaction, taking a document from `PENDING` to `READY`.

Two providers sit behind one protocol:

| Provider | Default | Needs | Vectors |
|----------|---------|-------|---------|
| `fake` | yes | nothing | deterministic, hash-derived, **no semantics** |
| `sentence_transformers` | no | `pip install -e '.[embeddings]'` | a real local model |

`fake` is the default so a base install and the entire test suite run with no model
download, no network, and no GPU. Its vectors are unit-length and stable — identical text
always gives the identical vector — but similar sentences do *not* land near each other. That
is deliberate: a test asserting retrieval quality against them would be measuring nothing, so
the provider is obviously fake rather than plausibly wrong.

**A provider's width is checked against `EMBEDDING_DIMENSIONS` at construction.** A pgvector
column has a fixed dimension and its ANN index is built for that width, so a 384-dimensional
model is refused at startup with a message naming both widths, rather than failing as an
opaque driver error midway through a batch. `ERDIC_EMBEDDING_MODEL` is likewise validated at
startup, since an empty value in a `.env` file otherwise surfaces as a 500 on first ingest.

Queries and documents are embedded by separate methods because BGE and E5 models are trained
with asymmetric instruction prefixes; leaving that to call sites guarantees someone
eventually forgets, and the resulting mismatch degrades retrieval silently.

Duplicate avoidance operates at three levels:

- Chunks that already hold a vector are skipped, so re-running is cheap and an interrupted
  run resumes rather than starting over.
- Identical chunk text within a batch is encoded once. Boilerplate — headers, disclaimers,
  standard clauses — repeats heavily across an enterprise corpus, and the encoder is the
  expensive part.
- Re-embedding with a *different* provider is refused unless `replace=true`, because vectors
  from two models in one index make every similarity score meaningless.

Status advances to `READY` only when every chunk holds a vector, counted from the database
rather than from what the run wrote — so a document left partly embedded by an interrupted run
reports `EMBEDDED`, not `READY`. `READY` means retrievable, and a chunk without an embedding
is invisible to vector search.

`scripts/verify_ingestion.py` exercises the whole chain over real HTTP against a running
server, including with the real model.

## Retrieval

`POST /api/v1/search` runs the full pipeline by default:

```
dense ──┐
        ├─→ reciprocal rank fusion ─→ 20 candidates ─→ cross-encoder ─→ top 8
bm25 ───┘
```

`retriever` selects `hybrid` (the default), `dense`, or `bm25`. A real run, with the
sentence-transformers encoder and the cross-encoder enabled:

```
Q: What notice period is required to terminate the vendor agreement?
   weights={'dense': 0.65, 'bm25': 0.35} rrf_k=60 candidates=20
   stages: dense=3 bm25=1 fused=3 returned=3 reranked=True
   timings: {'dense': 55.1, 'bm25': 5.5, 'fusion': 0.03, 'rerank': 300.3}
   1. [legal.txt]      score=+7.5914  ranks={'dense': 1, 'bm25': 1}
      scores={'dense': 0.757, 'bm25': 3.841, 'rrf': 0.0164, 'rerank': 7.591}
   2. [finance.txt]    score=-11.2654 ranks={'dense': 3}
```

### Fusion

**RRF combines ranks, not scores.** This is the whole reason it is used: cosine similarity is
bounded in [-1, 1] and BM25 is unbounded and corpus-dependent, so adding or averaging them is
meaningless and normalising them requires knowing each distribution in advance. Rank 1 from
either retriever contributes the same amount, which is what makes the two comparable at all.

```
rrf(d) = Σ_r  weight_r / (k + rank_r(d))
```

`ERDIC_FUSION_DENSE_WEIGHT` and `ERDIC_FUSION_BM25_WEIGHT` default to **0.65 / 0.35**; only
their ratio matters, and either may be zero to A/B a single retriever. `ERDIC_FUSION_RRF_K`
defaults to 60, where the gap between ranks 1 and 2 is small next to the gap between appearing
and not appearing — which is what makes fusion robust to one retriever being confidently wrong.
All four are overridable per request via the `fusion` block.

Both retrievers are asked for `ERDIC_FUSION_CANDIDATES` (20) results rather than the final 8,
because fusion can only reward agreement it can see: if each leg returned 8, a chunk ranked 9th
by dense and 1st by BM25 would never receive its dense contribution.

One consequence worth knowing before tuning weights: **RRF rewards consensus, so a decisive
keyword match is not guaranteed to win.** A chunk ranked second by both legs scores
`0.65/62 + 0.35/62 = 0.016129`, beating one ranked first by BM25 and last by dense at
`0.35/61 + 0.65/63 = 0.016055`. That is intended, and there is a test pinning it.

### Reranking

A cross-encoder scores the query and a passage *together*, which neither earlier stage does —
dense compares two independently computed vectors and BM25 counts term overlap. That is why it
ranks better and why it only runs over the small fused candidate set.

`ERDIC_RERANKER` defaults to `none`, which keeps the fused order and reports `reranked: false`.
It does not invent scores: a reranker that reordered by a deterministic hash would look like
reranking while destroying the ordering fusion produced. `cross_encoder` runs a real model via
the `embeddings` extra, opt-in because it downloads weights on first use.

Its effect is not cosmetic. Given candidates where fusion ranked the correct passage **last**,
the cross-encoder moved it to first, scoring `+7.22` against `-11.21` for an irrelevant one.

### Persistence

`persist: true` records the query and its ranked results, returning `query_id`. Searching does
not write by default; an audit row should be a deliberate act.

Every stage's score is stored in `retrieval_results.scores`, not just the final one. Knowing a
chunk ended up third is far less useful than knowing BM25 ranked it first, dense ranked it
ninth, fusion put it second, and the cross-encoder moved it to third — that trail is what turns
"the ranking looks wrong" into a diagnosis. `strategy` records which stage decided the rank
(`RERANK` when a cross-encoder did), and the fusion weights, `rrf_k`, reranker, and stage counts
are stored on the query, because a ranking is only interpretable next to the configuration that
produced it.

`scripts/verify_hybrid.py` exercises the whole pipeline over real HTTP.

### Dense (`retriever: "dense"`)

Embeds the question and returns the nearest chunks by cosine similarity.

| Supported | Detail |
|-----------|--------|
| `top_k` | defaults to `ERDIC_RETRIEVAL_TOP_K`, capped at 200; an over-large value is rejected, not clamped |
| similarity | cosine similarity, `1.0` identical; the raw pgvector distance is returned alongside it |
| document filtering | by document id, and by status |
| metadata filtering | JSONB containment on document *and* chunk metadata, served by the GIN indexes |
| page filtering | by 1-based page number |
| threshold | `min_similarity`, applied after ranking |

Returned per hit: chunk id and text, document id and title, page, section, token count,
character offsets, media type, both metadata objects, the similarity and distance, and a
prebuilt `citation_label` such as `report · Risks > Currency · p.7`.

Three decisions worth knowing:

**Scores are reported as similarity, not distance.** pgvector's `<=>` returns distance, where
smaller is better, which inverts every intuition about a "score" and breaks any threshold
written the obvious way. The conversion happens once, in the retriever.

**Only `READY` documents are searched by default.** READY is defined as "every chunk
embedded", so a partly embedded document would return an arbitrary subset of itself while
looking like a complete answer. The filter can be widened explicitly.

**The query is embedded with `embed_query`, never `embed_documents`.** BGE and E5 apply a
different instruction prefix to queries, and using the passage form degrades ranking quietly.

The `<=>` operator is used specifically because the HNSW index is built with
`vector_cosine_ops`. Verified on a 5 000-row table: `<=>` plans as
`Index Scan using ix_document_chunks_embedding_hnsw`, while `<->` falls back to a sequential
scan over the whole corpus.

One ANN caveat that is counter-intuitive: PostgreSQL applies a `WHERE` clause *after* the
index returns candidates, so a selective filter can yield fewer than `top_k` hits even when
enough matching rows exist. `hnsw.ef_search` is raised (`ERDIC_RETRIEVAL_EF_SEARCH_FILTERED`,
default 200) when filters are present, and set `LOCAL` so it cannot leak across a pooled
connection. The response also reports `searchable_chunks`, which distinguishes "nothing
matched the question" from "nothing matched the filters".

### Keyword (`retriever: "bm25"`)

Genuine Okapi BM25, not PostgreSQL's `ts_rank`. Those are different functions: `ts_rank` and
`ts_rank_cd` score term coverage and proximity with no notion of document frequency or length
normalisation, so ranking with them and calling it BM25 would be mislabelling.

```
score(q,d) = Σ  idf(t) · tf(t,d)·(k1+1) / ( tf(t,d) + k1·(1 − b + b·|d|/avgdl) )
idf(t)     = ln( 1 + (N − n(t) + 0.5) / (n(t) + 0.5) )
```

Every quantity comes from an inverted index in PostgreSQL: `chunk_terms(chunk_id, term,
term_frequency)` plus `document_chunks.lexeme_count`. Terms on both sides are produced by
`to_tsvector`, so the query and the index share PostgreSQL's stemmer and stopword list —
searching "terminating agreements" finds "terminates the agreement" because both reduce to
`termin` and `agreement`. `ERDIC_BM25_K1` and `ERDIC_BM25_B` default to the standard 1.2 and
0.75.

**Synchronisation with ingestion has two halves.** Removal is structural: `chunk_terms.chunk_id`
cascades, so deleting a chunk or re-chunking a document drops its terms in the same statement
and no application code can forget. Insertion happens in the same transaction that writes the
chunks, so there is no window in which a chunk exists but is invisible to keyword search, and
it is auditable afterwards — `lexeme_count IS NULL` marks a chunk that was never indexed, and
`BM25Indexer.find_unindexed` reports them.

Two other decisions:

**A chunk appears once however many query terms it matched.** The scoring query groups by
chunk; without that, a chunk containing three of the query's terms would come back three
times. Query terms are also de-duplicated, so repeating a word does not inflate its weight.

**IDF is computed corpus-wide, not over the filtered subset.** Document frequency is a
property of the collection; recomputing it per filter would make the same chunk score
differently depending on what else the caller happened to request.

`min_similarity` is refused for BM25 rather than applied: it is a cosine threshold, and BM25
scores are unbounded, so silently accepting it would hand back a result set the caller believes
was filtered. For the same reason a BM25 hit leaves `similarity` and `distance` null instead of
filling them with a number that is not one.

The filter predicates are part of a fixed SQL string, disabled by `NULL` checks rather than
appended conditionally, so no part of the statement is assembled from a runtime value.

## LLM layer

One provider protocol with two operations -- free-text `generate` and schema-validated
`generate_structured` -- behind `ERDIC_LLM_PROVIDER`:

| Provider | Speaks to | Needs |
|----------|-----------|-------|
| `fake` (default) | nothing: deterministic canned output derived from the requested schema | nothing |
| `openai` | any OpenAI-compatible chat API (OpenAI, vLLM, Together, llama.cpp server) | `ERDIC_LLM_MODEL`, optionally `ERDIC_LLM_API_KEY` / `ERDIC_LLM_BASE_URL` |
| `ollama` | Ollama's native API | `ERDIC_LLM_MODEL`, a running Ollama |
| `huggingface` | a local transformers model in-process | the `llm` extra |

Running against a real local model, which needs no credentials:

```bash
brew install ollama && ollama serve &          # or any existing Ollama server
ollama pull qwen2.5:7b-instruct
export ERDIC_LLM_PROVIDER=ollama ERDIC_LLM_MODEL=qwen2.5:7b-instruct
export ERDIC_LLM_MAX_TOKENS=2048               # small models narrate; a cut-off JSON reply fails
export ERDIC_EMBEDDING_PROVIDER=sentence_transformers ERDIC_RERANKER=cross_encoder
```

Model choice is load-bearing for grounded answering, and was measured rather than assumed:
`llama3.2:3b` answers correctly but returns an **empty citation list**, so citation
validation rejects its answers and the workflow refuses them -- correct behaviour, useless
output. A 7B instruct model copies chunk ids and verbatim quotes reliably. In Docker, point
`ERDIC_LLM_BASE_URL` at `http://host.docker.internal:11434` to use a host Ollama.

Structured output never trusts the server: whatever mode requested it
(`json_schema`, `json_object`, or prompt-embedded schema, via
`ERDIC_LLM_STRUCTURED_MODE`), the reply is parsed and validated locally against the
Pydantic schema. Prompts are versioned in a registry (`classify_query:v1`,
`generate_answer:v2`, ...) and the version that produced an answer is recorded with it.
The fake provider is honest about being fake: its answers fail citation validation by
design, so nothing downstream can mistake canned output for a grounded answer.

## Agent workflow

`POST /api/v1/query` runs a LangGraph state machine over twelve nodes:

```
query_understanding → query_classification ─┬→ query_decomposition ─┐
                                            └──────────────────────┴→ retrieval_strategy
  → hybrid_retrieval → reranking ─┬→ tool_execution → answer_generation
                                  │         (evidence sufficient)      │
                                  └→ query_reformulation ←─────────────┤
                                        (insufficient; loops back      │
                                         to hybrid_retrieval)          ↓
                     citation_validation → evidence_verification → finalize
                       (invalid: regenerate or reformulate)  (unsupported: reformulate)
```

Classification runs through a three-tier router -- the fine-tuned adapter when
`ERDIC_ROUTER_ADAPTER_DIR` points at one, the LLM prompt otherwise, and a deterministic
heuristic when the model is unreachable. Insufficient evidence or failed verification
triggers query reformulation and re-retrieval; invalid citations trigger regeneration. All
retry kinds share one budget (`ERDIC_AGENT_MAX_RETRIES`, default 2), so the worst case
stays bounded and an exhausted budget produces an honest "no relevant evidence" answer,
never an ungrounded one. Tools (a safe AST-restricted calculator that never executes
arbitrary code, and a corpus metadata lookup) run before generation and their outputs are
handed to the model as exact values.

## Evaluation

A native evaluation framework (RAGAS/DeepEval definitions, deliberately not their
libraries -- see [docs/evaluation.md](docs/evaluation.md)) measures faithfulness, answer
relevance, context precision/recall, citation correctness, hallucination rate, refusal
correctness, latencies, and token usage over a 56-question golden dataset spanning eight
question types. `python -m evaluation.run` compares a deliberately weakened baseline
(dense-only, single-pass) against the full pipeline and writes JSON, CSV, and Markdown
reports. A metric that cannot be measured reports *not applicable* with a reason, never a
fabricated zero.

The stored run in `artifacts/evaluation/` is real: 56 records with real embeddings, a real
reranker, and `ollama:qwen2.5:7b-instruct` generating and judging. The improved
configuration wins on every quality metric, most decisively on context precision
(0.25 → 0.76) and refusal correctness (0.29 → 0.77), for ~5.8× the latency. Full table,
counts, and the three recorded failures are in [docs/evaluation.md](docs/evaluation.md).

## Fine-tuned query router

`scripts/train_router.py` fine-tunes a small classifier over ten routing classes with
LoRA (QLoRA on CUDA machines with bitsandbytes; detected, and the mode that actually ran
is recorded in the manifest). `scripts/evaluate_router.py` compares it against the
deterministic heuristic baseline.

Measured honestly, on two test sets: the *generated* split scores 1.000 accuracy, but it
shares templates with training, so that is template recall. On a curated held-out set of
100 hand-authored questions with no shared template, the same adapter scores **0.48
accuracy / 0.4843 macro F1** against the heuristic's **0.27 / 0.2638** — a real gain
(+0.21) at a far lower absolute level. Detail, confusion analysis, and the leakage
measurement are in [docs/fine-tuning.md](docs/fine-tuning.md).

## Observability

Structured logs carry a request id propagated from FastAPI through LangGraph, with a
redaction filter that keeps API keys, passwords, and tokens out of every record. The
public `/api/v1/metrics` endpoint exposes Prometheus text format from a small hand-rolled
registry: request counts and latency histograms by route template, error counts by code,
retrieval and LLM call counts, token totals, workflow outcomes and retries, and the most
recent in-process evaluation scores. MLflow tracking is opt-in via `ERDIC_MLFLOW_*`
settings; the application runs identically with it disabled.

## Frontend

A Streamlit UI (`frontend/`, the `frontend` extra) that is a pure client of the HTTP
API: Research Chat with citations, confidence, and workflow detail; document management;
a retrieval inspector showing every stage's score per chunk; and evaluation/fine-tuning
dashboards that read the stored artifacts and the MLflow store. Missing artifacts render
as the command that produces them -- the dashboards never display fabricated numbers.

## Error contract

Every failure returns one shape, including framework errors such as routing 404s and
unhandled exceptions, so a client needs a single parser:

```json
{
  "error": {
    "code": "not_found",
    "message": "document missing",
    "details": {"id": "doc-1"},
    "request_id": "9f2c..."
  }
}
```

`code` is a stable machine-readable string; branch on it rather than on `message`.
`request_id` also comes back in the `X-Request-ID` header and appears in every log line
for that request, so quoting it is enough to find the trace.

Domain code raises `ErdicError` subclasses from `erdic.core.errors`; the translation to
HTTP lives in `erdic.api.errors`. Outside `ERDIC_ENV=local`, internal exception text is
withheld from responses because it can carry connection strings or record contents.

## Repositories

`erdic.repositories` is the only place that builds queries against the models, so a schema
change has one place to land. There is one repository per table, all sharing a generic
`BaseRepository` for `get`/`get_or_raise`/`list`/`count`/`add`/`delete`.

**No repository commits.** Writes `flush()` so generated ids and server defaults are
available, and the commit belongs to whoever owns the unit of work:

```python
with session_scope() as session:  # commits, or rolls back
    document = DocumentRepository(session).create(title=..., content_hash=...)
    DocumentChunkRepository(session).create_many([...])  # same transaction
```

If repositories committed, a failure while writing chunks would leave a document with no
chunks: visible, retrievable, and unciteable.

Two other conventions worth knowing. Listings are always totally ordered and capped at
`MAX_PAGE_SIZE`; an over-large `limit` is rejected rather than silently clamped, because a
caller that asked for 10 000 rows and got 200 without being told would draw wrong
conclusions. And validation that the database also enforces is duplicated in the
repository where it improves the message — a wrong-width embedding is named as such
instead of surfacing as an opaque driver error mid-transaction.

## Testing

```bash
make test              # everything
make test-unit         # unit only
make test-nodb         # everything that needs no PostgreSQL server
make cov               # with coverage
```

Tests strip `ERDIC_*` from the environment so results never depend on local
configuration. Database tests run against a real PostgreSQL server, because pgvector
behaviour cannot be faked; they skip with an actionable message when none is reachable.
`ERDIC_TEST_DATABASE_URL` selects the target, and the suite refuses to run unless the
database name marks it as disposable.
