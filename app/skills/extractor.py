"""Entity and key fact extraction skill.

Uses regex patterns and heuristics to extract structured information
from text without external API keys.
"""

from __future__ import annotations

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 5]


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extract named entities from text using regex patterns and heuristics.

    Extracts: dates, monetary values, percentages, proper nouns,
    emails, URLs, and organizations.

    Args:
        text: Input text to extract entities from.

    Returns:
        Dictionary mapping entity types to lists of extracted entities.
    """
    entities: dict[str, list[str]] = {
        "dates": [],
        "monetary_values": [],
        "percentages": [],
        "proper_nouns": [],
        "emails": [],
        "urls": [],
        "organizations": [],
    }

    # Dates: various formats
    date_patterns = [
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",  # MM/DD/YYYY, DD-MM-YYYY
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b",
        r"\bQ[1-4] \d{4}\b",  # Q1 2024
        r"\b\d{4}\b",  # Standalone year (4 digits)
    ]
    for pattern in date_patterns:
        matches = re.findall(pattern, text)
        entities["dates"].extend(matches)

    # Monetary values
    money_patterns = [
        r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|K))?\b",
        r"\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s*(?:dollars|USD|EUR|GBP)\b",
        r"\b(?:USD|EUR|GBP|JPY)\s*[\d,]+(?:\.\d{2})?\b",
    ]
    for pattern in money_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        entities["monetary_values"].extend(matches)

    # Percentages
    pct_pattern = r"\b\d+(?:\.\d+)?%|\b\d+(?:\.\d+)?\s*percent\b"
    entities["percentages"] = re.findall(pct_pattern, text, re.IGNORECASE)

    # Emails
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    entities["emails"] = re.findall(email_pattern, text)

    # URLs
    url_pattern = r"https?://[^\s<>\"']+|www\.[^\s<>\"']+"
    entities["urls"] = re.findall(url_pattern, text)

    # Proper nouns (capitalized sequences, excluding sentence starts)
    sentences = _split_sentences(text)
    proper_nouns: set[str] = set()
    for sent in sentences:
        # Find capitalized words that are not at the start of a sentence
        words = sent.split()
        for i, word in enumerate(words):
            if i == 0:
                continue
            # Multi-word proper nouns
            if re.match(r"^[A-Z][a-z]+$", word):
                proper_nouns.add(word)
        # Also find multi-word capitalized sequences
        multi_cap = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)+)\b", sent)
        proper_nouns.update(multi_cap)
    entities["proper_nouns"] = sorted(proper_nouns)[:20]

    # Organizations (words followed by Inc, Corp, Ltd, LLC, etc.)
    org_pattern = r"\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s+(?:Inc|Corp|Corporation|Ltd|LLC|Co|Company|Group|Partners|Associates|Foundation|Institute|University)\b"
    entities["organizations"] = re.findall(org_pattern, text)

    # Deduplicate all lists
    for key in entities:
        entities[key] = sorted(set(entities[key]))

    return entities


def extract_key_facts(text: str, top_n: int = 10) -> list[str]:
    """Identify key factual statements from text.

    Focuses on sentences containing numbers, proper nouns, or definitive verbs.

    Args:
        text: Input text.
        top_n: Maximum number of facts to return.

    Returns:
        List of key factual sentences.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    scored: list[tuple[float, str]] = []
    for sent in sentences:
        score = 0.0

        # Strong boost for numbers (data/facts)
        numbers = re.findall(r"\d+", sent)
        score += len(numbers) * 1.5

        # Boost for percentages
        if "%" in sent or "percent" in sent.lower():
            score += 2.0

        # Boost for monetary values
        if "$" in sent or any(w in sent.lower() for w in ["million", "billion", "revenue", "profit"]):
            score += 1.5

        # Boost for proper nouns (indicates specific entities)
        proper = re.findall(r"\b[A-Z][a-z]+\b", sent)
        # Exclude first word of sentence
        words = sent.split()
        if words and words[0] in proper:
            proper = proper[1:]
        score += len(proper) * 0.5

        # Boost for definitive verbs
        definitive = ["announced", "reported", "increased", "decreased", "launched",
                      "acquired", "released", "confirmed", "signed", "achieved",
                      "completed", "published", "discovered", "established"]
        for verb in definitive:
            if verb in sent.lower():
                score += 1.0
                break

        # Slight boost for longer sentences (more informative)
        word_count = len(sent.split())
        if 10 <= word_count <= 40:
            score += 0.5

        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in scored[:top_n]]


def extract_topics(chunks: list[str], top_n: int = 10) -> list[dict[str, float]]:
    """Identify top topic keywords across chunks using TF-IDF-like scoring.

    Args:
        chunks: List of text chunks.
        top_n: Number of top topics to return.

    Returns:
        List of dicts with 'term' and 'score' keys.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "between", "out", "off", "over", "under",
        "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "both", "few", "more", "most", "other", "some", "such", "no", "not",
        "only", "same", "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "about", "this", "that", "these", "those",
        "it", "its", "they", "them", "their", "he", "she", "his", "her", "we",
        "our", "you", "your", "i", "me", "my", "which", "what", "who", "also",
    }

    if not chunks:
        return []

    # Compute term frequency across all chunks
    total_tf: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    num_docs = len(chunks)

    for chunk in chunks:
        tokens = _tokenize(chunk)
        filtered = [t for t in tokens if t not in stopwords and len(t) > 2]
        total_tf.update(filtered)
        # Document frequency
        doc_freq.update(set(filtered))

    # TF-IDF-like scoring
    import math

    scored_terms: list[tuple[str, float]] = []
    for term, tf in total_tf.items():
        df = doc_freq[term]
        idf = math.log((num_docs + 1) / (df + 1)) + 1
        score = tf * idf
        scored_terms.append((term, score))

    scored_terms.sort(key=lambda x: x[1], reverse=True)
    return [{"term": term, "score": round(score, 3)} for term, score in scored_terms[:top_n]]
