"""Tests for skill-related API endpoints (summarize, compare, extract, conversation)."""

import os
from datetime import UTC, datetime

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


def _create_multi_doc_project(user_id: str = "u1") -> tuple[str, list[str]]:
    """Helper: create project with multiple indexed documents."""
    import os
    import tempfile

    project = store.create_project(user_id, "multi-doc-project")
    pid = project.id

    docs = [
        (
            "file-s1",
            "finance_report.txt",
            (
                "Financial Performance Q4 2024\n\n"
                "Revenue increased by 35 percent this quarter. "
                "Net profit reached $8.5 million, up from $6.2 million last quarter. "
                "Operating expenses were reduced by 10 percent through automation. "
                "The company expanded into three new markets in Southeast Asia.\n\n"
                "Employee retention improved to 95 percent."
            ),
        ),
        (
            "file-s2",
            "tech_overview.txt",
            (
                "Technology Trends Overview 2024\n\n"
                "Artificial intelligence adoption accelerated across all sectors. "
                "Cloud computing revenue grew by 28 percent year-over-year. "
                "Cybersecurity spending increased to $180 billion globally. "
                "Quantum computing made breakthroughs in error correction.\n\n"
                "Open source software powers 90 percent of modern applications."
            ),
        ),
    ]

    file_ids: list[str] = []
    all_chunk_texts: list[str] = []

    os.makedirs("data/blobs", exist_ok=True)

    for fid, filename, text in docs:
        # Write to a real temp file so summarize endpoint can parse it
        local_path = os.path.join("data/blobs", f"{fid}-{filename}")
        with open(local_path, "w") as f:
            f.write(text)

        fr = FileRecord(
            id=fid,
            owner_id=user_id,
            project_id=pid,
            filename=filename,
            size=len(text),
            mime_type="text/plain",
            upload_timestamp=datetime.now(UTC),
            parsed=True,
            local_path=local_path,
        )
        store.upsert_file(fr)
        file_ids.append(fid)

        chunks = chunk_document(fid, filename, fr.upload_timestamp, text)
        store.add_chunks(pid, chunks)
        all_chunk_texts.extend(c.text for c in chunks)

    refit(all_chunk_texts)
    store.reindex_embeddings(pid)

    return pid, file_ids


# ---------- Summarize ----------


def test_summarize_success() -> None:
    """POST /api/summarize returns a meaningful summary."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/summarize",
        json={"projectId": pid, "maxSentences": 5},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"]) > 20
    assert "keyPoints" in data
    assert isinstance(data["keyPoints"], list)
    assert len(data["keyPoints"]) > 0


def test_summarize_no_files() -> None:
    """POST /api/summarize on empty project returns 404."""
    project = store.create_project("u1", "empty-proj")
    client = TestClient(app)

    resp = client.post(
        "/api/summarize",
        json={"projectId": project.id, "maxSentences": 5},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 404


def test_summarize_with_file_filter() -> None:
    """POST /api/summarize with fileIds filters to specific files."""
    pid, file_ids = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/summarize",
        json={"projectId": pid, "fileIds": [file_ids[0]], "maxSentences": 3},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data


# ---------- Compare ----------


def test_compare_two_documents() -> None:
    """POST /api/compare returns comparison of two documents."""
    pid, file_ids = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/compare",
        json={"projectId": pid, "fileIds": file_ids},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "sharedThemes" in data
    assert "documents" in data
    assert len(data["documents"]) == 2
    assert "overlapScore" in data
    assert "contradictions" in data


def test_compare_needs_two_files() -> None:
    """POST /api/compare rejects requests with fewer than 2 files."""
    pid, file_ids = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/compare",
        json={"projectId": pid, "fileIds": [file_ids[0]]},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 422


# ---------- Extract ----------


def test_extract_entities() -> None:
    """POST /api/extract returns entity extraction results."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/extract",
        json={"projectId": pid, "extractionType": "entities"},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["extractionType"] == "entities"
    assert "entities" in data
    entities = data["entities"]
    assert "percentages" in entities
    assert "monetary_values" in entities


def test_extract_facts() -> None:
    """POST /api/extract returns extracted facts."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/extract",
        json={"projectId": pid, "extractionType": "facts"},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["extractionType"] == "facts"
    assert isinstance(data["facts"], list)
    assert len(data["facts"]) > 0


def test_extract_topics() -> None:
    """POST /api/extract returns extracted topics."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/extract",
        json={"projectId": pid, "extractionType": "topics"},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["extractionType"] == "topics"
    assert isinstance(data["topics"], list)
    assert len(data["topics"]) > 0
    assert "term" in data["topics"][0]
    assert "score" in data["topics"][0]


# ---------- Conversation ----------


def test_conversation_start() -> None:
    """POST /api/conversation/start creates a new conversation."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "conversationId" in data
    assert data["projectId"] == pid


def test_conversation_message() -> None:
    """POST /api/conversation/message returns answer in conversation context."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    # Start conversation
    start_resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    conv_id = start_resp.json()["conversationId"]

    # Send message
    resp = client.post(
        "/api/conversation/message",
        json={"conversationId": conv_id, "message": "What was the revenue growth?"},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert data["conversationId"] == conv_id


def test_conversation_history() -> None:
    """GET /api/conversation/{id}/history returns all turns."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    # Start and send message
    start_resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    conv_id = start_resp.json()["conversationId"]

    client.post(
        "/api/conversation/message",
        json={"conversationId": conv_id, "message": "Tell me about revenue"},
        headers={"x-user-id": "u1"},
    )

    # Get history
    resp = client.get(
        f"/api/conversation/{conv_id}/history",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "turns" in data
    assert len(data["turns"]) == 2
    assert data["turns"][0]["role"] == "user"
    assert data["turns"][1]["role"] == "assistant"


def test_conversation_not_found() -> None:
    """GET /api/conversation/{id}/history returns 404 for invalid ID."""
    client = TestClient(app)
    resp = client.get(
        "/api/conversation/nonexistent/history",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 404


def test_conversation_access_denied() -> None:
    """Conversation message from wrong user returns 403."""
    pid, _ = _create_multi_doc_project()
    client = TestClient(app)

    start_resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    conv_id = start_resp.json()["conversationId"]

    resp = client.post(
        "/api/conversation/message",
        json={"conversationId": conv_id, "message": "Hello"},
        headers={"x-user-id": "u2"},
    )
    assert resp.status_code == 403
