# Deep Research Webapp (Enterprise-Oriented Baseline)

This repository now includes a functional full-stack baseline:
1. Architecture and provider selection (`docs/architecture.md`)
2. OpenAPI contract (`openapi.yaml`)
3. Frontend wireframes (`docs/wireframes.md`)
4. Backend ingestion/query service (`app/`) with role-aware controls and ingestion job status
5. Interactive web UI on `/` with upload + query workflow and social/contact links

## Key improvements
- Project sharing roles (`read`, `query`, `write`, `owner`) in the in-memory access model
- File type + size validation during upload
- Ingestion job tracking endpoint (`/api/ingest/jobs/{jobId}`)
- Audit retrieval endpoint for owners (`/api/audit/retrievals`)
- Social profile links integrated into UI and API (`/api/me/profiles`)

## Run
```bash
pip install -e .[dev]
pytest
uvicorn app.main:app --reload
```
