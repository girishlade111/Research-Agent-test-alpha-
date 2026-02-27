# Deep Research Webapp Architecture (Enterprise Recommendation)

## Selected provider stack
- **Frontend:** Next.js + React + TypeScript + TailwindCSS.
- **API Gateway/App:** FastAPI (Python) for orchestration and auth-bound APIs.
- **Async Ingestion Workers:** Celery/RQ workers running parse → chunk → embed jobs with retries.
- **Object Storage:** S3-compatible bucket with SSE-KMS.
- **Metadata DB:** PostgreSQL.
- **Vector DB:** Weaviate (hybrid sparse+dense retrieval with metadata filters).
- **Sparse Search:** OpenSearch/Elasticsearch BM25.
- **Cache/Queue:** Redis.
- **LLM + Embeddings:** OpenAI (primary) + local Ollama fallback.
- **OCR/Vision:** AWS Textract or Tesseract + vision model fallback.
- **Observability:** OpenTelemetry + Prometheus + Grafana.

## Diagram
```mermaid
flowchart LR
    U[User Browser] --> FE[Research UI + Social Profile Bar]
    FE --> API[FastAPI API Gateway]
    API --> AUTH[Auth Service / OAuth]
    API --> PG[(PostgreSQL)]
    API --> S3[(S3 Object Storage)]
    API --> Q[Redis Queue]
    Q --> W[Ingestion Worker]
    W --> PARSE[PDF/DOCX/XLSX/OCR Parsers]
    PARSE --> EMB[Embedding Provider]
    EMB --> VDB[(Weaviate / Vector DB)]
    API --> RET[Retrieval Orchestrator]
    RET --> VDB
    RET --> BM25[(OpenSearch)]
    RET --> WEB[Web Search Connector]
    RET --> LLM[LLM Provider]
    API --> OBS[OTel Metrics/Logs]
```

## Security controls
- Per-user and per-project RBAC enforced at API + retrieval filter level.
- TLS in transit; SSE-KMS encryption for object storage and encrypted DB volumes.
- Privacy toggles for web augmentation; redactable retrieval/query logs.
- GDPR delete workflow removes blobs + metadata + vectors + logs.
- Ingestion audit + retrieval audit endpoints for enterprise debugging/compliance.
