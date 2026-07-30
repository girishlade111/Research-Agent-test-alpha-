"""Tests for agent skills: summarization, comparison, extraction, conversation, web search."""

from datetime import UTC, datetime

from httpx import ASGITransport, AsyncClient

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


def _create_project_with_files() -> tuple[str, list[str]]:
    """Helper: create a project with 2 indexed files."""
    project = store.create_project("u1", "test-project")
    pid = project.id

    texts = [
        (
            "report_q4.txt",
            "Financial Report Q4 2024.\n\n"
            "Revenue increased by 22 percent in Q4 2024. "
            "The company reported $5.2 million in net profit. "
            "Growth was driven by expansion into European markets. "
            "Risk remains supply chain volatility and rising costs.\n\n"
            "Total headcount grew from 150 to 200 employees. "
            "The board approved a new share buyback program worth $10 million.",
        ),
        (
            "market_analysis.txt",
            "Market Analysis Report 2024.\n\n"
            "The technology sector saw a decline in Q4 due to regulatory pressure. "
            "Revenue decreased by 5 percent across the industry. "
            "Key players include Microsoft Corp, Apple Inc, and Google LLC.\n\n"
            "Consumer spending shifted toward AI products. "
            "The European market showed growth of 12 percent year-over-year.",
        ),
    ]

    file_ids: list[str] = []
    all_chunks_texts: list[str] = []

    for i, (filename, text) in enumerate(texts):
        fid = f"file-{i}"
        fr = FileRecord(
            id=fid,
            owner_id="u1",
            project_id=pid,
            filename=filename,
            size=len(text),
            mime_type="text/plain",
            upload_timestamp=datetime.now(UTC),
            parsed=True,
            local_path="",
        )
        store.upsert_file(fr)
        file_ids.append(fid)

        chunks = chunk_document(fid, filename, fr.upload_timestamp, text)
        store.add_chunks(pid, chunks)
        all_chunks_texts.extend(c.text for c in chunks)

    # Refit embeddings with all chunks
    refit(all_chunks_texts)
    store.reindex_embeddings(pid)

    return pid, file_ids


# ---------- Summarization ----------


def test_summarize_endpoint() -> None:
    """POST /api/summarize returns a summary with key points."""
    pid, file_ids = _create_project_with_files()

    from starlette.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/api/summarize",
        json={"projectId": pid, "maxSentences": 5},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert len(data["summary"]) > 0
    assert "keyPoints" in data
    assert isinstance(data["keyPoints"], list)


def test_summarize_requires_access() -> None:
    """POST /api/summarize enforces project access."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/api/summarize",
        json={"projectId": pid, "maxSentences": 5},
        headers={"x-user-id": "unknown-user"},
    )
    assert resp.status_code == 403


# ---------- Compare ----------


def test_compare_endpoint() -> None:
    """POST /api/compare returns structured comparison for 2+ docs."""
    pid, file_ids = _create_project_with_files()

    from starlette.testclient import TestClient

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


def test_compare_requires_two_files() -> None:
    """POST /api/compare rejects requests with < 2 files."""
    pid, file_ids = _create_project_with_files()

    from starlette.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/api/compare",
        json={"projectId": pid, "fileIds": [file_ids[0]]},
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 422  # pydantic validation min_length=2


# ---------- Extract ----------


def test_extract_entities() -> None:
    """POST /api/extract returns entities from documents."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

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
    # Should find percentages and monetary values from our test docs
    entities = data["entities"]
    assert isinstance(entities["percentages"], list)
    assert isinstance(entities["monetary_values"], list)


def test_extract_facts() -> None:
    """POST /api/extract returns key facts."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

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
    """POST /api/extract returns top topics."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

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
    # Each topic should have term and score
    assert "term" in data["topics"][0]
    assert "score" in data["topics"][0]


def test_extract_requires_access() -> None:
    """POST /api/extract enforces project access."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/api/extract",
        json={"projectId": pid, "extractionType": "entities"},
        headers={"x-user-id": "unauthorized"},
    )
    assert resp.status_code == 403


# ---------- Conversation ----------


def test_conversation_start() -> None:
    """POST /api/conversation/start creates a conversation."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

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
    """POST /api/conversation/message returns an answer with context."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

    client = TestClient(app)

    # Start conversation
    start_resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    conv_id = start_resp.json()["conversationId"]

    # Send message
    msg_resp = client.post(
        "/api/conversation/message",
        json={"conversationId": conv_id, "message": "What was the revenue growth?"},
        headers={"x-user-id": "u1"},
    )
    assert msg_resp.status_code == 200
    data = msg_resp.json()
    assert "answer" in data
    assert "sources" in data
    assert data["conversationId"] == conv_id


def test_conversation_history() -> None:
    """GET /api/conversation/{id}/history returns turns."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

    client = TestClient(app)

    # Start + send message
    start_resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    conv_id = start_resp.json()["conversationId"]

    client.post(
        "/api/conversation/message",
        json={"conversationId": conv_id, "message": "Tell me about the report"},
        headers={"x-user-id": "u1"},
    )

    # Get history
    hist_resp = client.get(
        f"/api/conversation/{conv_id}/history",
        headers={"x-user-id": "u1"},
    )
    assert hist_resp.status_code == 200
    data = hist_resp.json()
    assert "turns" in data
    assert len(data["turns"]) == 2  # user + assistant
    assert data["turns"][0]["role"] == "user"
    assert data["turns"][1]["role"] == "assistant"


def test_conversation_not_found() -> None:
    """GET /api/conversation/{id}/history returns 404 for invalid ID."""
    from starlette.testclient import TestClient

    client = TestClient(app)
    resp = client.get(
        "/api/conversation/nonexistent/history",
        headers={"x-user-id": "u1"},
    )
    assert resp.status_code == 404


def test_conversation_access_denied() -> None:
    """Conversation endpoints deny access to wrong user."""
    pid, _ = _create_project_with_files()

    from starlette.testclient import TestClient

    client = TestClient(app)

    start_resp = client.post(
        "/api/conversation/start",
        json={"projectId": pid},
        headers={"x-user-id": "u1"},
    )
    conv_id = start_resp.json()["conversationId"]

    # Different user tries to send message
    msg_resp = client.post(
        "/api/conversation/message",
        json={"conversationId": conv_id, "message": "Hello"},
        headers={"x-user-id": "u2"},
    )
    assert msg_resp.status_code == 403


# ---------- Unit tests for skills modules ----------


def test_summarizer_module() -> None:
    """Test summarizer module directly."""
    from app.skills.summarizer import extract_key_points, summarize_chunks, summarize_document

    text = (
        "The economy grew by 3.5 percent last year. "
        "Inflation remained at 2 percent. "
        "Unemployment dropped to 4 percent. "
        "Consumer spending increased significantly. "
        "Housing prices rose by 8 percent across major cities."
    )
    summary = summarize_document(text, max_sentences=3)
    assert len(summary) > 0
    assert isinstance(summary, str)

    chunks = [text, "Additional data shows strong exports."]
    chunk_summary = summarize_chunks(chunks, query="economy growth", max_sentences=2)
    assert len(chunk_summary) > 0

    points = extract_key_points(text, max_points=3)
    assert len(points) > 0
    assert len(points) <= 3


def test_extractor_module() -> None:
    """Test extractor module directly."""
    from app.skills.extractor import extract_entities, extract_key_facts, extract_topics

    text = (
        "On January 15, 2024, Microsoft Corp announced revenue of $50 billion. "
        "Growth was 22% year-over-year. Contact info@example.com for details. "
        "Visit https://example.com for more information."
    )
    entities = extract_entities(text)
    assert len(entities["monetary_values"]) > 0
    assert len(entities["percentages"]) > 0
    assert len(entities["emails"]) > 0
    assert len(entities["urls"]) > 0

    facts = extract_key_facts(text, top_n=5)
    assert len(facts) > 0

    chunks = [text, "AI market grew to $150 billion in 2024."]
    topics = extract_topics(chunks, top_n=5)
    assert len(topics) > 0


def test_comparator_module() -> None:
    """Test comparator module directly."""
    from app.skills.comparator import compare_documents, find_contradictions

    docs = [
        ("doc1.txt", "Revenue increased by 20 percent due to strong sales in Q4. Market share grew."),
        ("doc2.txt", "Revenue decreased by 5 percent due to regulatory issues. Market competition intensified."),
    ]
    result = compare_documents(docs)
    assert "sharedThemes" in result
    assert "documents" in result
    assert result["totalDocuments"] == 2

    contradictions = find_contradictions(
        ["Revenue increased by 20 percent in Q4."],
        ["Revenue decreased by 5 percent in Q4."],
    )
    assert isinstance(contradictions, list)


def test_conversation_module() -> None:
    """Test conversation module directly."""
    from app.skills.conversation import ConversationManager, rewrite_query

    mgr = ConversationManager()
    cid = mgr.start_conversation("user1", "proj1")
    assert cid

    mgr.add_turn(cid, "user", "Tell me about Microsoft revenue")
    mgr.add_turn(cid, "assistant", "Microsoft reported $50 billion in revenue.")

    ctx = mgr.get_context(cid, max_turns=10)
    assert len(ctx) == 2

    history = mgr.get_history(cid)
    assert len(history) == 2
    assert history[0]["role"] == "user"

    # Test query rewriting with pronouns
    rewritten = rewrite_query("What about it?", ctx)
    # Should expand the pronoun using context
    assert len(rewritten) > len("What about it?")


def test_web_search_module() -> None:
    """Test web search module structure (not actual network call)."""
    from app.skills.web_search import SearchResult, WebSearchProvider, format_web_results

    # Test formatting
    results = [
        SearchResult(title="Test Result", url="https://example.com", snippet="A test snippet"),
        SearchResult(title="Another Result", url="https://example.org", snippet="More info"),
    ]
    formatted = format_web_results(results)
    assert "Test Result" in formatted
    assert "https://example.com" in formatted
    assert "[1]" in formatted
    assert "[2]" in formatted

    # Test empty results
    assert "No web results found" in format_web_results([])
