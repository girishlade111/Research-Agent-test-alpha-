"""Tests for query/retrieval API endpoint."""

import io
from datetime import UTC, datetime
from unittest.mock import patch

from starlette.testclient import TestClient

from app.embeddings import get_vectorizer_store, refit
from app.main import app, conversation_manager, store
from app.models import FileRecord
from app.pipeline import chunk_document
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


def _create_ingested_project(user_id: str = "u1") -> tuple[str, str]:
    """Helper: create project with an ingested document. Returns (project_id, file_id)."""
    project = store.create_project(user_id, "query-test-project")
    pid = project.id

    text = (
        "Artificial Intelligence Report 2024\n\n"
        "Machine learning models have significantly improved in performance. "
        "Large language models demonstrate emergent reasoning capabilities. "
        "The global AI market is valued at approximately $200 billion.\n\n"
        "Deep learning architectures power modern NLP systems. "
        "Transformer models have revolutionized text understanding."
    )

    fid = "file-query-1"
    fr = FileRecord(
        id=fid,
        owner_id=user_id,
        project_id=pid,
        filename="ai_report.txt",
        size=len(text),
        mime_type="text/plain",
        upload_timestamp=datetime.now(UTC),
        parsed=True,
        local_path="",
    )
    store.upsert_file(fr)

    chunks = chunk_document(fid, "ai_report.txt", fr.upload_timestamp, text)
    store.add_chunks(pid, chunks)

    all_texts = [c.text for c in chunks]
    refit(all_texts)
    store.reindex_embeddings(pid)

    return pid, fid


def test_query_with_results() -> None:
    """POST /api/query returns answer and sources for matching query."""
    pid, fid = _create_ingested_project()
    client = TestClient(app)

    resp = client.post(
        "/api/query",
        json={
            "projectId": pid,
            "userId": "u1",
            "query": "What is the AI market value?",
            "topK": 5,
            "useWeb": False,
            "filters": {},
        },
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answerText" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert "confidence" in data
    assert "followups" in data


def test_query_no_results() -> None:
    """POST /api/query returns no-info answer when nothing matches."""
    # Create project with no documents
    project = store.create_project("u1", "empty-project")
    client = TestClient(app)

    resp = client.post(
        "/api/query",
        json={
            "projectId": project.id,
            "userId": "u1",
            "query": "What about quantum computing?",
            "topK": 5,
            "useWeb": False,
            "filters": {},
        },
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answerText" in data
    # When no chunks match, should indicate lack of info
    assert "don't have enough" in data["answerText"].lower() or len(data["sources"]) == 0


def test_query_with_file_filter() -> None:
    """POST /api/query with fileIds filter only searches selected files."""
    pid, fid = _create_ingested_project()

    # Add a second file
    text2 = "Biology Report\n\nCells divide through mitosis. DNA encodes genetic information."
    fid2 = "file-query-2"
    fr2 = FileRecord(
        id=fid2,
        owner_id="u1",
        project_id=pid,
        filename="biology.txt",
        size=len(text2),
        mime_type="text/plain",
        upload_timestamp=datetime.now(UTC),
        parsed=True,
        local_path="",
    )
    store.upsert_file(fr2)
    chunks2 = chunk_document(fid2, "biology.txt", fr2.upload_timestamp, text2)
    store.add_chunks(pid, chunks2)

    # Refit with all chunks
    all_texts = store.get_all_chunk_texts(pid)
    refit(all_texts)
    store.reindex_embeddings(pid)

    client = TestClient(app)

    # Query with file filter for only file 2
    resp = client.post(
        "/api/query",
        json={
            "projectId": pid,
            "userId": "u1",
            "query": "Tell me about AI",
            "topK": 5,
            "useWeb": False,
            "filters": {"fileIds": [fid2]},
        },
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # All sources should come from the filtered file
    for source in data["sources"]:
        assert source["fileId"] == fid2


def test_query_user_mismatch() -> None:
    """POST /api/query returns 403 when x-user-id differs from body userId."""
    pid, _ = _create_ingested_project()
    client = TestClient(app)

    resp = client.post(
        "/api/query",
        json={
            "projectId": pid,
            "userId": "u1",
            "query": "some question",
            "topK": 5,
            "useWeb": False,
            "filters": {},
        },
        headers={"x-user-id": "different-user"},
    )
    assert resp.status_code == 403


def test_query_no_access() -> None:
    """POST /api/query returns 403 for user without project access."""
    pid, _ = _create_ingested_project(user_id="owner1")
    client = TestClient(app)

    resp = client.post(
        "/api/query",
        json={
            "projectId": pid,
            "userId": "hacker",
            "query": "steal data",
            "topK": 5,
            "useWeb": False,
            "filters": {},
        },
        headers={"x-user-id": "hacker"},
    )
    assert resp.status_code == 403


def test_query_with_web_enabled() -> None:
    """POST /api/query with web enabled includes web results (mocked)."""
    pid, _ = _create_ingested_project()
    client = TestClient(app)

    # Enable web for user
    store.settings["u1"] = {"allowWeb": True}

    # Mock web search to avoid real HTTP calls
    mock_html = (
        '<a class="result__a" href="https://example.com/ai">AI News</a>'
        '<a class="result__snippet">Latest AI developments</a>'
    )
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.text = mock_html
        mock_post.return_value.raise_for_status = lambda: None

        resp = client.post(
            "/api/query",
            json={
                "projectId": pid,
                "userId": "u1",
                "query": "AI market trends",
                "topK": 5,
                "useWeb": True,
                "filters": {},
            },
            headers={"x-user-id": "u1"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "sources" in data


def test_query_with_conversation_context() -> None:
    """POST /api/query with conversationId uses conversation history."""
    pid, _ = _create_ingested_project()
    client = TestClient(app)

    # Start a conversation
    conv_resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    conv_id = conv_resp.json()["conversationId"]

    # Send a message to build context
    client.post(
        "/api/conversation/message",
        json={"conversationId": conv_id, "message": "Tell me about AI market value"},
        headers={"x-user-id": "u1"},
    )

    # Query with conversation context
    resp = client.post(
        "/api/query",
        json={
            "projectId": pid,
            "userId": "u1",
            "query": "What about that topic?",
            "topK": 5,
            "useWeb": False,
            "filters": {},
            "conversationId": conv_id,
        },
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answerText" in data


def test_query_empty_query_rejected() -> None:
    """POST /api/query rejects whitespace-only queries."""
    pid, _ = _create_ingested_project()
    client = TestClient(app)

    resp = client.post(
        "/api/query",
        json={
            "projectId": pid,
            "userId": "u1",
            "query": "   ",
            "topK": 5,
            "useWeb": False,
            "filters": {},
        },
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 422


def test_audit_retrievals() -> None:
    """GET /api/audit/retrievals returns retrieval logs."""
    pid, _ = _create_ingested_project()
    client = TestClient(app)

    # Perform a query first to generate a log
    client.post(
        "/api/query",
        json={
            "projectId": pid,
            "userId": "u1",
            "query": "AI market value",
            "topK": 5,
            "useWeb": False,
            "filters": {},
        },
        headers={"x-user-id": "u1"},
    )

    # Get audit logs
    resp = client.get(
        f"/api/audit/retrievals?projectId={pid}",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["query"] == "AI market value"
