"""Tests for project CRUD and sharing API endpoints."""

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


def test_create_project_success() -> None:
    """POST /api/projects creates a project and returns projectId."""
    client = TestClient(app)
    resp = client.post(
        "/api/projects",
        json={"name": "My Research Project"},
        headers={"x-user-id": "user1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "projectId" in data
    assert data["name"] == "My Research Project"
    # Verify project exists in store
    assert data["projectId"] in store.projects


def test_create_project_missing_header() -> None:
    """POST /api/projects without x-user-id returns 422."""
    client = TestClient(app)
    resp = client.post(
        "/api/projects",
        json={"name": "Test Project"},
    )
    assert resp.status_code == 422


def test_create_project_short_name() -> None:
    """POST /api/projects with name < 2 chars returns 422."""
    client = TestClient(app)
    resp = client.post(
        "/api/projects",
        json={"name": "X"},
        headers={"x-user-id": "user1"},
    )
    assert resp.status_code == 422


def test_create_project_invalid_name_characters() -> None:
    """POST /api/projects with dangerous characters returns 422."""
    client = TestClient(app)
    resp = client.post(
        "/api/projects",
        json={"name": "<script>alert('xss')</script>"},
        headers={"x-user-id": "user1"},
    )
    assert resp.status_code == 422


def test_share_project_success() -> None:
    """POST /api/project/{id}/share shares with another user."""
    client = TestClient(app)
    # Create project first
    create_resp = client.post(
        "/api/projects",
        json={"name": "Shared Project"},
        headers={"x-user-id": "owner1"},
    )
    project_id = create_resp.json()["projectId"]

    # Share with another user
    resp = client.post(
        f"/api/project/{project_id}/share",
        json={"targetUserId": "collaborator1", "role": "query"},
        headers={"x-user-id": "owner1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "shared"
    assert data["targetUserId"] == "collaborator1"
    assert data["role"] == "query"


def test_share_project_not_owner() -> None:
    """POST /api/project/{id}/share by non-owner returns 403."""
    client = TestClient(app)
    # Create project
    create_resp = client.post(
        "/api/projects",
        json={"name": "Owner Only"},
        headers={"x-user-id": "owner1"},
    )
    project_id = create_resp.json()["projectId"]

    # Share project first to give another user access
    client.post(
        f"/api/project/{project_id}/share",
        json={"targetUserId": "user2", "role": "write"},
        headers={"x-user-id": "owner1"},
    )

    # Non-owner tries to share
    resp = client.post(
        f"/api/project/{project_id}/share",
        json={"targetUserId": "user3", "role": "query"},
        headers={"x-user-id": "user2"},
    )
    assert resp.status_code == 403


def test_share_project_not_found() -> None:
    """POST /api/project/{id}/share for nonexistent project returns 404."""
    client = TestClient(app)
    resp = client.post(
        "/api/project/nonexistent-id/share",
        json={"targetUserId": "user2", "role": "query"},
        headers={"x-user-id": "owner1"},
    )
    assert resp.status_code == 404


def test_health_endpoint() -> None:
    """GET /api/health returns status ok."""
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "deep-research-api"


def test_profiles_endpoint() -> None:
    """GET /api/me/profiles returns social links."""
    client = TestClient(app)
    resp = client.get("/api/me/profiles")
    assert resp.status_code == 200
    data = resp.json()
    assert "github" in data
    assert "linkedin" in data
