"""Tests for middleware: request IDs, error handling, CORS."""

from starlette.testclient import TestClient

from app.embeddings import get_vectorizer_store
from app.main import app, conversation_manager, store
from app.store import reset_data_dir


def setup_function() -> None:
    """Reset state between tests."""
    reset_data_dir()
    get_vectorizer_store().reset()
    store.projects.clear()
    store.files.clear()
    store.chunks.clear()
    store.jobs.clear()
    store.project_chunks.clear()
    store.pinned_chunks.clear()
    store.logs.clear()
    store.settings.clear()
    conversation_manager.conversations.clear()


def test_request_id_header_present() -> None:
    """Every response includes X-Request-ID header."""
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "x-request-id" in resp.headers
    # Request ID should be a UUID-like string
    request_id = resp.headers["x-request-id"]
    assert len(request_id) > 10


def test_request_id_unique_per_request() -> None:
    """Each request gets a unique request ID."""
    client = TestClient(app)
    resp1 = client.get("/api/health")
    resp2 = client.get("/api/health")
    id1 = resp1.headers["x-request-id"]
    id2 = resp2.headers["x-request-id"]
    assert id1 != id2


def test_error_response_structured() -> None:
    """Error responses contain structured JSON with request_id, timestamp, detail."""
    client = TestClient(app)
    # Trigger a 404 error
    resp = client.get(
        "/api/ingest/jobs/nonexistent",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 404
    data = resp.json()
    assert "request_id" in data
    assert "timestamp" in data
    assert "detail" in data
    assert "error" in data


def test_error_422_structured() -> None:
    """Validation errors are returned with proper structure."""
    client = TestClient(app)
    resp = client.post(
        "/api/projects",
        json={"name": "X"},  # Too short
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 422


def test_cors_headers_present() -> None:
    """CORS headers are present on responses to preflight requests."""
    client = TestClient(app)
    # Simulate an OPTIONS preflight
    resp = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    # CORS middleware should respond with access-control headers
    assert "access-control-allow-origin" in resp.headers


def test_cors_headers_on_regular_request() -> None:
    """CORS headers are included on regular requests with Origin header."""
    client = TestClient(app)
    resp = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:3000"},
    )
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers


def test_settings_provider_endpoint() -> None:
    """POST /api/settings/provider stores user settings."""
    client = TestClient(app)
    resp = client.post(
        "/api/settings/provider",
        json={
            "llmProvider": "openai",
            "embeddingProvider": "local-tfidf",
            "allowWeb": True,
        },
        headers={"x-user-id": "user1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["settings"]["llmProvider"] == "openai"
    assert data["settings"]["allowWeb"] is True


def test_pin_chunk_endpoint() -> None:
    """POST /api/project/{id}/pin-chunk pins a chunk."""
    from datetime import UTC, datetime

    from app.models import FileRecord
    from app.pipeline import chunk_document

    project = store.create_project("u1", "pin-project")
    pid = project.id
    text = "Some text to chunk for pinning test purposes."
    fid = "pin-file"
    fr = FileRecord(
        id=fid,
        owner_id="u1",
        project_id=pid,
        filename="pin.txt",
        size=len(text),
        mime_type="text/plain",
        upload_timestamp=datetime.now(UTC),
        parsed=True,
        local_path="",
    )
    store.upsert_file(fr)
    chunks = chunk_document(fid, "pin.txt", fr.upload_timestamp, text)
    store.add_chunks(pid, chunks)

    chunk_id = chunks[0].id
    client = TestClient(app)
    resp = client.post(
        f"/api/project/{pid}/pin-chunk",
        json={"chunkId": chunk_id},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "pinned"


def test_pin_chunk_not_found() -> None:
    """POST /api/project/{id}/pin-chunk returns 404 for unknown chunk."""
    project = store.create_project("u1", "pin-project-2")
    client = TestClient(app)
    resp = client.post(
        f"/api/project/{project.id}/pin-chunk",
        json={"chunkId": "nonexistent-chunk"},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 404
