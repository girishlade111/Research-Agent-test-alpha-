"""Answer engine for synthesizing responses from retrieved chunks.

Provides extractive summarization, follow-up generation, and confidence scoring
without requiring external LLM API keys.
"""

from __future__ import annotations

import re
from typing import Any


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple heuristics."""
    # Split on sentence-ending punctuation followed by space or end of string
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()]


def _score_sentence(sentence: str, query_tokens: set[str]) -> float:
    """Score a sentence by relevance to query tokens."""
    sent_tokens = set(sentence.lower().split())
    if not sent_tokens:
        return 0.0
    overlap = len(query_tokens & sent_tokens)
    return overlap / max(len(query_tokens), 1)


def generate_answer(query: str, chunks: list[tuple[float, Any]], max_chunks: int = 5) -> str:
    """Generate an answer by extractive summarization from top retrieved chunks.

    Selects the most relevant sentences from the top chunks based on
    query-term overlap, then assembles them into a coherent answer.

    Args:
        query: The user's query string.
        chunks: List of (score, chunk) tuples, sorted by relevance descending.
        max_chunks: Maximum number of chunks to consider.

    Returns:
        A synthesized answer string.
    """
    if not chunks:
        return "I don't have enough information in the indexed corpus to answer this question."

    top_chunks = chunks[:max_chunks]
    query_tokens = set(query.lower().split())

    # Collect and score sentences from all top chunks
    scored_sentences: list[tuple[float, str, str]] = []  # (score, sentence, source_filename)
    seen_sentences: set[str] = set()

    for retrieval_score, chunk in top_chunks:
        sentences = _split_sentences(chunk.text)
        filename = chunk.metadata.get("filename", "unknown")

        for sentence in sentences:
            normalized = sentence.lower().strip()
            if normalized in seen_sentences or len(sentence) < 10:
                continue
            seen_sentences.add(normalized)

            relevance = _score_sentence(sentence, query_tokens)
            # Combine sentence relevance with chunk retrieval score
            combined_score = 0.6 * relevance + 0.4 * retrieval_score
            scored_sentences.append((combined_score, sentence, filename))

    if not scored_sentences:
        # Fallback: return snippet from top chunk
        return top_chunks[0][1].text[:300]

    # Sort by score and select top sentences
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    selected = scored_sentences[:5]

    # Build answer text
    answer_parts: list[str] = []
    answer_parts.append("Based on the indexed documents:\n")

    for i, (score, sentence, filename) in enumerate(selected, 1):
        answer_parts.append(f"- {sentence}")

    answer = "\n".join(answer_parts)

    # Add brief summary note
    sources_used = set(s[2] for s in selected)
    if sources_used:
        answer += f"\n\n(Derived from: {', '.join(sorted(sources_used))})"

    return answer


def generate_followups(query: str, chunks: list[tuple[float, Any]]) -> list[str]:
    """Generate follow-up question suggestions based on retrieved content.

    Analyzes the content of retrieved chunks to suggest related questions
    the user might want to ask.

    Args:
        query: The original query.
        chunks: List of (score, chunk) tuples.

    Returns:
        A list of suggested follow-up questions.
    """
    if not chunks:
        return [
            "Try uploading more documents to expand the knowledge base.",
            "Rephrase your question with different keywords.",
        ]

    followups: list[str] = []
    query_tokens = set(query.lower().split())

    # Collect unique topics/keywords from chunks that are NOT in the query
    chunk_keywords: set[str] = set()
    for _, chunk in chunks[:5]:
        words = chunk.text.lower().split()
        for word in words:
            clean = re.sub(r"[^a-z0-9]", "", word)
            if (
                clean
                and len(clean) > 3
                and clean not in query_tokens
                and clean not in {"this", "that", "with", "from", "have", "been", "were", "they", "their", "would", "could", "should"}
            ):
                chunk_keywords.add(clean)

    # Generate follow-ups based on discovered keywords
    keyword_list = sorted(chunk_keywords, key=len, reverse=True)[:6]

    if keyword_list:
        followups.append(f"Can you tell me more about {keyword_list[0]}?")
    if len(keyword_list) > 2:
        followups.append(f"How does {keyword_list[1]} relate to {keyword_list[2]}?")

    # Add generic but useful follow-ups
    followups.append("What are the key findings or conclusions?")
    followups.append("Are there any risks or concerns mentioned?")

    # Limit to 4 follow-ups
    return followups[:4]


def compute_confidence(scores: list[float]) -> float:
    """Compute a meaningful confidence score based on retrieval score distribution.

    Uses the distribution of top scores to estimate how confident the system
    is in its answer. High confidence means multiple chunks scored well with
    low variance; low confidence means sparse or inconsistent results.

    Args:
        scores: List of retrieval scores for top chunks.

    Returns:
        A confidence value between 0.0 and 1.0.
    """
    if not scores:
        return 0.0

    if len(scores) == 1:
        # Single score: confidence is the score itself, capped at 0.8
        return min(scores[0], 0.8)

    # Use mean of top scores as base confidence
    top_scores = sorted(scores, reverse=True)[:5]
    mean_score = sum(top_scores) / len(top_scores)

    # Penalize high variance (inconsistent results)
    if len(top_scores) > 1:
        variance = sum((s - mean_score) ** 2 for s in top_scores) / len(top_scores)
        std_dev = variance**0.5
        # Higher std_dev means less confidence
        consistency_bonus = max(0.0, 0.2 - std_dev)
    else:
        consistency_bonus = 0.0

    # More sources that scored well increases confidence
    coverage_bonus = min(0.1, len(top_scores) * 0.02)

    confidence = mean_score + consistency_bonus + coverage_bonus

    # Clamp to [0.0, 1.0]
    return round(max(0.0, min(1.0, confidence)), 3)
