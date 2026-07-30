"""Tests for file upload and ingestion API endpoints."""

import io
import os
from datetime import UTC, datetime

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


def _create_project(user_id: str = "u1", name: str = "test-project") -> str:
    """Helper to create a project and return its ID."""
    client = TestClient(app)
    resp = client.post(
        "/api/projects",
        json={"name": name},
        headers={"x-user-id": user_id},
    )
    return resp.json()["projectId"]


def test_upload_file_success() -> None:
    """POST /api/upload successfully uploads a text file."""
    project_id = _create_project()
    client = TestClient(app)

    file_content = b"This is a test document with some research content."
    resp = client.post(
        f"/api/upload?projectId={project_id}",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "fileId" in data
    assert "ingestionJobId" in data
    assert data["parseStatus"] == "pending"


def test_upload_unsupported_type() -> None:
    """POST /api/upload rejects unsupported file types."""
    project_id = _create_project()
    client = TestClient(app)

    file_content = b"binary data"
    resp = client.post(
        f"/api/upload?projectId={project_id}",
        files={"file": ("test.exe", io.BytesIO(file_content), "application/x-msdownload")},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 400


def test_upload_no_access() -> None:
    """POST /api/upload denies access to users without write permission."""
    project_id = _create_project(user_id="owner1")
    client = TestClient(app)

    file_content = b"Some content"
    resp = client.post(
        f"/api/upload?projectId={project_id}",
        files={"file": ("test.txt", io.BytesIO(file_content), "text/plain")},
        headers={"x-user-id": "unauthorized-user"},
    )
    assert resp.status_code == 403


def test_ingest_file_success() -> None:
    """POST /api/ingest/{file_id} processes and chunks a file."""
    project_id = _create_project()
    client = TestClient(app)

    # Upload file first
    file_content = (
        b"Machine Learning Report 2024\n\n"
        b"Neural networks have evolved significantly. "
        b"Transformer architectures dominate natural language processing. "
        b"Large language models show emergent capabilities.\n\n"
        b"Computer vision benefits from self-supervised learning approaches. "
        b"Reinforcement learning achieves superhuman performance in many domains."
    )
    upload_resp = client.post(
        f"/api/upload?projectId={project_id}",
        files={"file": ("research.txt", io.BytesIO(file_content), "text/plain")},
        headers={"x-user-id": "u1"},
    )
    file_id = upload_resp.json()["fileId"]

    # Ingest the file
    resp = client.post(
        f"/api/ingest/{file_id}",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["fileId"] == file_id
    assert data["chunks"] > 0
    assert "parseDetails" in data


def test_ingest_file_not_found() -> None:
    """POST /api/ingest/{file_id} returns 404 for missing file."""
    client = TestClient(app)
    resp = client.post(
        "/api/ingest/nonexistent-file",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 404


def test_ingest_job_status() -> None:
    """GET /api/ingest/jobs/{job_id} returns job status."""
    project_id = _create_project()
    client = TestClient(app)

    # Upload file
    file_content = b"Test content for job status check."
    upload_resp = client.post(
        f"/api/upload?projectId={project_id}",
        files={"file": ("status.txt", io.BytesIO(file_content), "text/plain")},
        headers={"x-user-id": "u1"},
    )
    job_id = upload_resp.json()["ingestionJobId"]

    # Check job status
    resp = client.get(
        f"/api/ingest/jobs/{job_id}",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["file_id"] == upload_resp.json()["fileId"]


def test_ingest_job_not_found() -> None:
    """GET /api/ingest/jobs/{job_id} returns 404 for missing job."""
    client = TestClient(app)
    resp = client.get(
        "/api/ingest/jobs/nonexistent-job",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 404


def test_upload_large_file() -> None:
    """POST /api/upload rejects files exceeding MAX_FILE_SIZE with 413."""
    project_id = _create_project()
    client = TestClient(app)

    # Create content larger than MAX_FILE_SIZE (25MB)
    # Use a small mock by temporarily setting max size very low
    from app.main import MAX_FILE_SIZE
    import app.main as main_module

    original_max = main_module.MAX_FILE_SIZE
    main_module.MAX_FILE_SIZE = 100  # Set to 100 bytes for test

    try:
        file_content = b"x" * 200  # Exceeds 100 bytes
        resp = client.post(
            f"/api/upload?projectId={project_id}",
            files={"file": ("big.txt", io.BytesIO(file_content), "text/plain")},
            headers={"x-user-id": "u1"},
        )
        assert resp.status_code == 413
    finally:
        main_module.MAX_FILE_SIZE = original_max


def test_list_files() -> None:
    """GET /api/files returns files for a project."""
    project_id = _create_project()
    client = TestClient(app)

    # Upload a file
    file_content = b"Document content here."
    client.post(
        f"/api/upload?projectId={project_id}",
        files={"file": ("doc.txt", io.BytesIO(file_content), "text/plain")},
        headers={"x-user-id": "u1"},
    )

    # List files
    resp = client.get(
        f"/api/files?projectId={project_id}",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["filename"] == "doc.txt"


def test_delete_file() -> None:
    """DELETE /api/files/{file_id} removes a file."""
    project_id = _create_project()
    client = TestClient(app)

    # Upload and ingest
    file_content = b"Content to delete."
    upload_resp = client.post(
        f"/api/upload?projectId={project_id}",
        files={"file": ("delete_me.txt", io.BytesIO(file_content), "text/plain")},
        headers={"x-user-id": "u1"},
    )
    file_id = upload_resp.json()["fileId"]

    # Delete
    resp = client.delete(
        f"/api/files/{file_id}",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"

    # Verify file is gone
    assert file_id not in store.files
