"""Tests for pipeline internals: chunking, parsing, and embedding quality."""

import os
import tempfile
from datetime import UTC, datetime

from app.embeddings import embed, get_vectorizer_store, refit
from app.pipeline import chunk_document, parse_content
from app.store import InMemoryStore, cosine_similarity, reset_data_dir


def setup_function() -> None:
    """Reset state between tests."""
    reset_data_dir()
    get_vectorizer_store().reset()


def test_chunk_document_overlap() -> None:
    """Verify chunks have overlapping content at boundaries."""
    text = (
        "First section content here with enough words to span multiple chunks. "
        "This continues with more detail about the research findings and methodology. "
        "Additional paragraphs provide context about the experimental setup.\n\n"
        "Second section discusses results. The performance improvements were significant. "
        "Metrics showed a 40 percent improvement over baseline. "
        "Further analysis reveals interesting patterns in the data distribution. "
        "The methodology was validated across multiple datasets and scenarios. "
        "Final conclusions draw connections between observations and theory. "
        "We recommend further investigation into these phenomena for future work. "
        "This is an extended passage designed to produce more than one chunk "
        "so that we can verify the overlapping behavior of the sliding window."
    )

    chunks = chunk_document("f1", "test.txt", datetime.now(UTC), text)

    # Should produce multiple chunks
    assert len(chunks) >= 2

    # Check overlap: end of chunk N should partially overlap with start of chunk N+1
    for i in range(len(chunks) - 1):
        current_end_text = chunks[i].text[-50:]  # last 50 chars
        next_start_text = chunks[i + 1].text[:50]  # first 50 chars

        # The overlap means at least some text should appear in both
        current_words = set(current_end_text.lower().split())
        next_words = set(next_start_text.lower().split())
        # At least some word overlap due to overlapping windows
        shared = current_words & next_words
        assert len(shared) > 0 or chunks[i].end_offset > chunks[i + 1].start_offset


def test_chunk_document_metadata() -> None:
    """Verify chunks contain correct metadata."""
    ts = datetime.now(UTC)
    text = "Hello world\n\nSecond paragraph content here."
    chunks = chunk_document("f1", "doc.txt", ts, text)

    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.file_id == "f1"
    assert chunk.metadata["filename"] == "doc.txt"
    assert chunk.metadata["page"] == 1
    assert chunk.metadata["charOffsets"][0] == chunk.start_offset
    assert chunk.metadata["charOffsets"][1] == chunk.end_offset
    assert len(chunk.embedding) > 0


def test_parse_content_text() -> None:
    """parse_content reads text files correctly."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document.\nWith multiple lines.")
        f.flush()
        path = f.name

    try:
        text, details = parse_content(path, "text/plain")
        assert "test document" in text
        assert details["parser"] == "text"
        assert details["pages"] == 1
    finally:
        os.unlink(path)


def test_parse_content_csv() -> None:
    """parse_content handles CSV files with row counts."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,age,city\nAlice,30,NYC\nBob,25,LA\n")
        f.flush()
        path = f.name

    try:
        text, details = parse_content(path, "text/csv")
        assert "Alice" in text
        assert details["parser"] == "text"
        assert details["rows"] == 2  # 3 rows total - 1 header = 2
    finally:
        os.unlink(path)


def test_parse_content_empty_file() -> None:
    """parse_content handles empty files gracefully."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        f.flush()
        path = f.name

    try:
        text, details = parse_content(path, "text/plain")
        assert text == ""
        assert details["parser"] == "empty-file"
    finally:
        os.unlink(path)


def test_parse_content_missing_file() -> None:
    """parse_content handles missing files gracefully."""
    text, details = parse_content("/nonexistent/path/file.txt", "text/plain")
    assert text == ""
    assert details["parser"] == "missing-file"


def test_embed_similar_texts_score_higher() -> None:
    """Similar texts should have higher cosine similarity than dissimilar texts."""
    # Build a corpus for the vectorizer
    corpus = [
        "Machine learning algorithms process data to find patterns",
        "Deep learning neural networks learn representations from data",
        "Cooking recipes require ingredients and careful preparation",
        "The weather forecast predicts rain tomorrow afternoon",
        "Natural language processing understands human text",
        "Artificial intelligence systems automate complex tasks",
    ]
    refit(corpus)

    # Similar texts (both about ML/AI)
    emb_ml1 = embed("Machine learning uses algorithms to learn from data")
    emb_ml2 = embed("Deep learning models train on large datasets")

    # Dissimilar text
    emb_cook = embed("Baking a chocolate cake requires flour and eggs")

    sim_same_topic = cosine_similarity(emb_ml1, emb_ml2)
    sim_diff_topic = cosine_similarity(emb_ml1, emb_cook)

    # Similar texts should score higher
    assert sim_same_topic > sim_diff_topic


def test_embed_different_texts_score_lower() -> None:
    """Completely different topics should have low similarity."""
    corpus = [
        "Programming in Python is popular for data science",
        "JavaScript powers web applications and frameworks",
        "Gardening tips for growing tomatoes in summer",
        "The history of ancient Roman architecture",
        "Quantum mechanics explains subatomic particle behavior",
        "Classical music compositions by Beethoven and Mozart",
    ]
    refit(corpus)

    emb_tech = embed("Python programming for software development")
    emb_garden = embed("Growing vegetables in a backyard garden")

    sim = cosine_similarity(emb_tech, emb_garden)
    # Should be relatively low (close to 0 or negative is fine)
    assert sim < 0.5


def test_chunk_empty_text() -> None:
    """chunk_document returns empty list for empty text."""
    chunks = chunk_document("f1", "empty.txt", datetime.now(UTC), "")
    assert chunks == []

    chunks = chunk_document("f1", "whitespace.txt", datetime.now(UTC), "   \n\n   ")
    assert chunks == []
