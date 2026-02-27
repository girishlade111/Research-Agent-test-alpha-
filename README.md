# Deep Research Webapp (Baseline)

This repository delivers the requested phase-1 artifacts:
1. Architecture and provider selection (`docs/architecture.md`)
2. OpenAPI contract (`openapi.yaml`)
3. Frontend wireframes (`docs/wireframes.md`)
4. Backend ingestion/query service (`app/`) with E2E tests (`tests/test_e2e.py`)

## Run
```bash
pip install -e .[dev]
pytest
uvicorn app.main:app --reload
```
