"""Extractive summarization using a TextRank-like algorithm.

Builds a sentence similarity graph and picks top sentences by centrality.
No external API keys required.
"""

from __future__ import annotations

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer: lowercase, split on non-alpha."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if len(s.strip()) > 10]
    return sentences


def _sentence_similarity(s1_tokens: list[str], s2_tokens: list[str]) -> float:
    """Compute Jaccard-like similarity between two tokenized sentences."""
    if not s1_tokens or not s2_tokens:
        return 0.0
    set1 = set(s1_tokens)
    set2 = set(s2_tokens)
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return 0.0
    return len(intersection) / len(union)


def _textrank_scores(sentences: list[str], damping: float = 0.85, iterations: int = 30) -> list[float]:
    """Compute TextRank centrality scores for sentences.

    Uses a similarity graph where edges are weighted by sentence overlap.
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # Tokenize all sentences
    tokenized = [_tokenize(s) for s in sentences]

    # Build similarity matrix
    sim_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            score = _sentence_similarity(tokenized[i], tokenized[j])
            sim_matrix[i][j] = score
            sim_matrix[j][i] = score

    # Normalize outgoing weights
    out_weights = [sum(sim_matrix[i]) for i in range(n)]

    # PageRank-style iteration
    scores = [1.0 / n] * n
    for _ in range(iterations):
        new_scores = [0.0] * n
        for i in range(n):
            rank_sum = 0.0
            for j in range(n):
                if i != j and out_weights[j] > 0:
                    rank_sum += (sim_matrix[j][i] / out_weights[j]) * scores[j]
            new_scores[i] = (1 - damping) / n + damping * rank_sum
        scores = new_scores

    return scores


def summarize_document(text: str, max_sentences: int = 10) -> str:
    """Produce an extractive summary of a document using TextRank.

    Args:
        text: The full document text.
        max_sentences: Maximum number of sentences to include.

    Returns:
        A summary string composed of the top-ranked sentences in original order.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return text[:500] if text else ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    scores = _textrank_scores(sentences)

    # Get indices of top sentences by score
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in indexed[:max_sentences]])

    return " ".join(sentences[i] for i in top_indices)


def summarize_chunks(chunks: list[str], query: str | None = None, max_sentences: int = 5) -> str:
    """Summarize a list of text chunks, optionally focusing on a query.

    Args:
        chunks: List of text chunks to summarize.
        query: Optional query to boost relevant sentences.
        max_sentences: Maximum sentences in summary.

    Returns:
        An extractive summary string.
    """
    combined = " ".join(chunks)
    sentences = _split_sentences(combined)
    if not sentences:
        return combined[:500] if combined else ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    scores = _textrank_scores(sentences)

    # If a query is provided, boost sentences containing query terms
    if query:
        query_tokens = set(_tokenize(query))
        for i, sent in enumerate(sentences):
            sent_tokens = set(_tokenize(sent))
            overlap = len(query_tokens & sent_tokens)
            if overlap > 0:
                scores[i] += 0.3 * (overlap / max(len(query_tokens), 1))

    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in indexed[:max_sentences]])

    return " ".join(sentences[i] for i in top_indices)


def extract_key_points(text: str, max_points: int = 5) -> list[str]:
    """Extract key points (important sentences) from a text.

    Focuses on sentences that contain numbers, proper nouns, or strong verbs.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    scored: list[tuple[float, str]] = []
    for sent in sentences:
        score = 0.0
        # Boost sentences with numbers
        if re.search(r"\d+", sent):
            score += 1.0
        # Boost sentences with capitalized words (proper nouns)
        caps = re.findall(r"\b[A-Z][a-z]+\b", sent)
        score += len(caps) * 0.3
        # Boost longer sentences (more informative)
        words = sent.split()
        if len(words) > 8:
            score += 0.5
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:max_points]]
