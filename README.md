# Deep Research Webapp — Enterprise-Oriented Baseline

A full-stack **deep research agent** baseline: a FastAPI backend with a pluggable multi-skill
pipeline (extraction, summarization, comparison, conversation, web search), role-aware access
controls, ingestion job tracking, and an interactive web UI on `/` with upload + query workflow.

## Key Features

- **Multi-skill research pipeline** (`app/pipeline.py` + `app/skills/`): extract, summarize, compare,
  converse, and web-search skills orchestrated per query
- **Ingestion service** with file type + size validation, job status tracking
  (`/api/ingest/jobs/{jobId}`)
- **Role-aware access model**: `read`, `query`, `write`, `owner` project roles (in-memory)
- **Audit retrieval endpoint** for owners (`/api/audit/retrievals`)
- **Social profile links** integrated into UI and API (`/api/me/profiles`)
- **Rate limiting middleware** (slowapi) and structured logging
- **OpenAPI contract** (`openapi.yaml`) + architecture docs (`docs/architecture.md`) and UI
  wireframes (`docs/wireframes.md`)
- **97-test suite** covering API endpoints, skills, pipeline, and middleware

## Tech Stack

Python 3.11+, FastAPI, Uvicorn, Pydantic v2, scikit-learn, numpy, slowapi, pytest, httpx

## Quick Start

```bash
pip install -e .[dev]
pytest
uvicorn app.main:app --reload
```

Then open http://localhost:8000 for the interactive web UI.

## Project Structure

```
app/
  main.py            # FastAPI application entrypoint
  config.py          # Settings (pydantic-settings)
  pipeline.py        # Research pipeline orchestration
  skills/            # extractor, summarizer, comparator, conversation, web_search
  models.py          # Pydantic models
  store.py           # In-memory access model + project roles
  middleware.py      # Request middleware
  rate_limiter.py    # slowapi rate limiting
  embeddings.py      # Embedding helpers
docs/
  architecture.md    # Provider selection + architecture notes
  wireframes.md      # Frontend wireframes
tests/               # 97 tests (API, skills, pipeline, middleware)
openapi.yaml         # Full OpenAPI contract
pyproject.toml       # Package metadata + dependencies
```

## Environment Variables

No secrets required to run the baseline locally. Provider keys (if a web-search or LLM
provider skill is enabled later) go in a local `.env` — never committed.

## Deploy Notes

The backend is a standard ASGI app (FastAPI + Uvicorn) with upload, job-tracking, and
rate-limited API endpoints. It needs a long-running Python server process, so it does not
deploy to static hosting (GitHub Pages / Cloudflare Pages). Suitable targets: a VPS,
Fly.io, Render, or Railway.

---

Built by **Girish Lade** — https://ladestack.in
