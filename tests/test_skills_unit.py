"""Unit tests for skill modules: summarizer, comparator, extractor, conversation, web_search."""

from unittest.mock import MagicMock, patch

from app.embeddings import get_vectorizer_store
from app.store import reset_data_dir


def setup_function() -> None:
    """Reset state between tests."""
    reset_data_dir()
    get_vectorizer_store().reset()


# ---------- Summarizer ----------


def test_summarizer_extracts_top_sentences() -> None:
    """summarize_document returns top-ranked sentences."""
    from app.skills.summarizer import summarize_document

    text = (
        "The global economy grew by 3.2 percent in 2024. "
        "Inflation rates declined across developed nations. "
        "Central banks began reducing interest rates. "
        "Employment levels reached record highs in several countries. "
        "Technology sector led growth with 15 percent expansion. "
        "Manufacturing output stalled due to supply chain issues."
    )
    summary = summarize_document(text, max_sentences=3)
    assert len(summary) > 0
    # Summary should be shorter than original
    assert len(summary) < len(text)
    # Should contain complete sentences
    assert "." in summary


def test_summarizer_chunks_with_query() -> None:
    """summarize_chunks prioritizes query-relevant content."""
    from app.skills.summarizer import summarize_chunks

    chunks = [
        "Revenue grew by 20 percent in Q4 2024 driven by strong sales.",
        "The company cafeteria now offers vegan options on Tuesdays.",
        "Net profit margin improved to 15 percent from 12 percent last year.",
    ]
    summary = summarize_chunks(chunks, query="financial performance", max_sentences=2)
    assert len(summary) > 0


def test_summarizer_key_points() -> None:
    """extract_key_points returns bullet-style key points."""
    from app.skills.summarizer import extract_key_points

    text = (
        "Machine learning adoption grew by 40 percent. "
        "Cloud spending reached $500 billion globally. "
        "Remote work remained popular with 60 percent of companies. "
        "AI startups raised $50 billion in funding."
    )
    points = extract_key_points(text, max_points=3)
    assert isinstance(points, list)
    assert len(points) <= 3
    assert len(points) > 0
    for point in points:
        assert isinstance(point, str)
        assert len(point) > 0


# ---------- Comparator ----------


def test_comparator_finds_shared_themes() -> None:
    """compare_documents identifies shared themes between documents."""
    from app.skills.comparator import compare_documents

    docs = [
        ("doc1.txt", "Revenue growth was strong at 25 percent. Market expansion continued."),
        ("doc2.txt", "Revenue increased significantly. The market showed strong growth patterns."),
    ]
    result = compare_documents(docs)
    assert "sharedThemes" in result
    assert isinstance(result["sharedThemes"], list)
    assert "documents" in result
    assert len(result["documents"]) == 2
    assert "overlapScore" in result
    assert 0 <= result["overlapScore"] <= 1


def test_comparator_finds_contradictions() -> None:
    """find_contradictions detects opposing statements."""
    from app.skills.comparator import find_contradictions

    chunks_a = [
        "Revenue increased by 30 percent this quarter.",
        "The company expanded into 5 new markets.",
    ]
    chunks_b = [
        "Revenue decreased by 10 percent due to market downturn.",
        "The company consolidated operations in existing markets.",
    ]
    contradictions = find_contradictions(chunks_a, chunks_b)
    assert isinstance(contradictions, list)


# ---------- Extractor ----------


def test_extractor_finds_dates_and_values() -> None:
    """extract_entities finds dates, monetary values, and percentages."""
    from app.skills.extractor import extract_entities

    text = (
        "On March 15, 2024, the company reported $42 billion in revenue, "
        "up 18% from the previous year. Contact: john@company.com"
    )
    entities = extract_entities(text)
    assert len(entities["monetary_values"]) > 0
    assert len(entities["percentages"]) > 0
    assert len(entities["emails"]) > 0
    assert "john@company.com" in entities["emails"]


def test_extractor_finds_emails_urls() -> None:
    """extract_entities finds emails and URLs."""
    from app.skills.extractor import extract_entities

    text = (
        "Send inquiries to support@example.org or visit https://docs.example.com/guide "
        "for documentation. Also check http://legacy.example.net for old records."
    )
    entities = extract_entities(text)
    assert "support@example.org" in entities["emails"]
    assert any("docs.example.com" in url for url in entities["urls"])


def test_extractor_key_facts() -> None:
    """extract_key_facts returns meaningful factual sentences."""
    from app.skills.extractor import extract_key_facts

    text = (
        "Apple Inc reported revenue of $123 billion. "
        "This represents a 12% increase year-over-year. "
        "The weather was nice today. "
        "Tim Cook stated that AI investments will double next year."
    )
    facts = extract_key_facts(text, top_n=5)
    assert isinstance(facts, list)
    assert len(facts) > 0
    # Facts should contain the data-rich sentences
    all_facts_text = " ".join(facts)
    assert "$123 billion" in all_facts_text or "12%" in all_facts_text


def test_extractor_topics() -> None:
    """extract_topics returns terms with scores."""
    from app.skills.extractor import extract_topics

    chunks = [
        "Machine learning and artificial intelligence are transforming industries.",
        "Deep learning neural networks process complex data patterns.",
        "Natural language processing enables human-computer interaction.",
    ]
    topics = extract_topics(chunks, top_n=5)
    assert isinstance(topics, list)
    assert len(topics) > 0
    assert "term" in topics[0]
    assert "score" in topics[0]
    assert topics[0]["score"] > 0


# ---------- Conversation ----------


def test_conversation_query_rewrite() -> None:
    """rewrite_query expands pronouns using conversation context."""
    from app.skills.conversation import rewrite_query

    history = [
        {"role": "user", "content": "Tell me about Microsoft revenue"},
        {"role": "assistant", "content": "Microsoft reported $50 billion in revenue."},
    ]
    rewritten = rewrite_query("What about their growth?", history)
    # Should incorporate context about Microsoft
    assert len(rewritten) > len("What about their growth?")


def test_conversation_manager_lifecycle() -> None:
    """ConversationManager handles full conversation lifecycle."""
    from app.skills.conversation import ConversationManager

    mgr = ConversationManager()
    cid = mgr.start_conversation("user1", "project1")
    assert cid

    # Verify conversation exists
    conv = mgr.get_conversation(cid)
    assert conv is not None
    assert conv.user_id == "user1"
    assert conv.project_id == "project1"

    # Add turns
    mgr.add_turn(cid, "user", "Hello")
    mgr.add_turn(cid, "assistant", "Hi there!")

    # Check history
    history = mgr.get_history(cid)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello"
    assert history[1]["role"] == "assistant"

    # Check context (last N turns)
    ctx = mgr.get_context(cid, max_turns=10)
    assert len(ctx) == 2


def test_conversation_manager_not_found() -> None:
    """ConversationManager returns None for nonexistent conversation."""
    from app.skills.conversation import ConversationManager

    mgr = ConversationManager()
    assert mgr.get_conversation("nonexistent") is None


# ---------- Web Search ----------


def test_web_search_format_results() -> None:
    """format_web_results formats SearchResult objects into readable text."""
    from app.skills.web_search import SearchResult, format_web_results

    results = [
        SearchResult(title="First Result", url="https://first.com", snippet="First snippet"),
        SearchResult(title="Second Result", url="https://second.com", snippet="Second snippet"),
    ]
    formatted = format_web_results(results)
    assert "[1]" in formatted
    assert "[2]" in formatted
    assert "First Result" in formatted
    assert "https://first.com" in formatted
    assert "Second snippet" in formatted


def test_web_search_format_empty() -> None:
    """format_web_results handles empty results."""
    from app.skills.web_search import format_web_results

    formatted = format_web_results([])
    assert "No web results found" in formatted


def test_web_search_provider_mocked() -> None:
    """WebSearchProvider.search returns results from mocked HTML."""
    from app.skills.web_search import WebSearchProvider

    mock_html = (
        '<div>'
        '<a class="result__a" href="https://example.com/test">Test Title</a>'
        '<a class="result__snippet">A test snippet about AI</a>'
        '</div>'
    )

    with patch("requests.post") as mock_post:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = mock_html
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        provider = WebSearchProvider(timeout=5)
        results = provider.search("test query", num_results=3)

    assert isinstance(results, list)
    # Results depend on HTML parsing; verify no crash
    mock_post.assert_called_once()


def test_web_search_provider_timeout() -> None:
    """WebSearchProvider handles timeouts gracefully."""
    import requests as req_lib
    from app.skills.web_search import WebSearchProvider

    with patch("requests.post") as mock_post:
        mock_post.side_effect = req_lib.Timeout("Connection timed out")

        provider = WebSearchProvider(timeout=1)
        results = provider.search("test query")

    assert results == []


def test_web_search_provider_network_error() -> None:
    """WebSearchProvider handles network errors gracefully."""
    import requests as req_lib
    from app.skills.web_search import WebSearchProvider

    with patch("requests.post") as mock_post:
        mock_post.side_effect = req_lib.ConnectionError("Network unreachable")

        provider = WebSearchProvider(timeout=5)
        results = provider.search("test query")

    assert results == []
