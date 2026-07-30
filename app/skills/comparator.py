"""Document comparison skill.

Identifies shared themes, unique points per document, and potential contradictions.
Uses TF-IDF term analysis to find key terms per document and compare overlap.
No external API keys required.
"""

from __future__ import annotations

import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_key_terms(text: str, top_n: int = 20) -> list[str]:
    """Extract top key terms from text based on frequency, excluding stopwords."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "out", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "each",
        "every", "both", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "but", "and", "or", "if", "while", "about", "this",
        "that", "these", "those", "it", "its", "they", "them", "their", "he",
        "she", "his", "her", "we", "our", "you", "your", "i", "me", "my",
        "which", "what", "who", "whom",
    }
    tokens = _tokenize(text)
    filtered = [t for t in tokens if t not in stopwords and len(t) > 2]
    counter = Counter(filtered)
    return [term for term, _ in counter.most_common(top_n)]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def compare_documents(doc_texts: list[tuple[str, str]], aspects: list[str] | None = None) -> dict:
    """Compare multiple documents and identify shared/unique themes.

    Args:
        doc_texts: List of (filename, text) tuples.
        aspects: Optional list of aspects to focus comparison on.

    Returns:
        A structured comparison with shared themes, unique points, and summary.
    """
    if len(doc_texts) < 2:
        return {"error": "At least 2 documents are required for comparison"}

    # Extract key terms per document
    doc_terms: dict[str, list[str]] = {}
    doc_term_sets: dict[str, set[str]] = {}
    for filename, text in doc_texts:
        terms = _get_key_terms(text, top_n=30)
        doc_terms[filename] = terms
        doc_term_sets[filename] = set(terms)

    # Find shared themes (terms appearing in 2+ documents)
    all_term_sets = list(doc_term_sets.values())
    filenames = [fn for fn, _ in doc_texts]
    shared_terms: set[str] = set()
    for i in range(len(all_term_sets)):
        for j in range(i + 1, len(all_term_sets)):
            shared_terms |= all_term_sets[i] & all_term_sets[j]

    # Find unique terms per document
    unique_per_doc: dict[str, list[str]] = {}
    for filename in filenames:
        others = set()
        for other_fn in filenames:
            if other_fn != filename:
                others |= doc_term_sets[other_fn]
        unique = doc_term_sets[filename] - others
        unique_per_doc[filename] = sorted(unique)[:10]

    # If aspects are provided, filter to relevant terms
    if aspects:
        aspect_tokens = set()
        for a in aspects:
            aspect_tokens.update(_tokenize(a))
        shared_terms = {t for t in shared_terms if t in aspect_tokens or any(a in t for a in aspect_tokens)}

    # Build document summaries
    doc_summaries: list[dict] = []
    for filename, text in doc_texts:
        sentences = _split_sentences(text)
        summary_sent = sentences[:3] if sentences else [text[:200]]
        doc_summaries.append({
            "filename": filename,
            "keyTerms": doc_terms[filename][:10],
            "uniquePoints": unique_per_doc.get(filename, []),
            "previewSentences": summary_sent,
        })

    return {
        "sharedThemes": sorted(shared_terms)[:15],
        "documents": doc_summaries,
        "overlapScore": round(len(shared_terms) / max(len(set().union(*all_term_sets)), 1), 3),
        "totalDocuments": len(doc_texts),
    }


def find_contradictions(chunks_a: list[str], chunks_b: list[str]) -> list[dict]:
    """Find potentially contradicting statements between two sets of chunks.

    Uses keyword opposition patterns (increase/decrease, higher/lower, etc.)
    to identify sentences that may contradict each other.

    Args:
        chunks_a: Text chunks from document A.
        chunks_b: Text chunks from document B.

    Returns:
        List of potential contradictions with source sentences.
    """
    opposites = [
        ("increase", "decrease"),
        ("higher", "lower"),
        ("more", "less"),
        ("growth", "decline"),
        ("positive", "negative"),
        ("improved", "worsened"),
        ("rose", "fell"),
        ("gain", "loss"),
        ("profit", "loss"),
        ("up", "down"),
        ("better", "worse"),
        ("success", "failure"),
        ("expand", "contract"),
        ("growing", "shrinking"),
    ]

    text_a = " ".join(chunks_a)
    text_b = " ".join(chunks_b)
    sentences_a = _split_sentences(text_a)
    sentences_b = _split_sentences(text_b)

    contradictions: list[dict] = []

    for word_a, word_b in opposites:
        # Find sentences in A containing word_a and sentences in B containing word_b
        matches_a = [s for s in sentences_a if word_a in s.lower()]
        matches_b = [s for s in sentences_b if word_b in s.lower()]

        for sa in matches_a:
            for sb in matches_b:
                # Check if they share a common topic (at least 2 common meaningful words)
                tokens_a = set(_tokenize(sa)) - {"the", "a", "is", "are", "was", "in", "of", "to"}
                tokens_b = set(_tokenize(sb)) - {"the", "a", "is", "are", "was", "in", "of", "to"}
                common = tokens_a & tokens_b - {word_a, word_b}
                if len(common) >= 2:
                    contradictions.append({
                        "statementA": sa,
                        "statementB": sb,
                        "oppositionPattern": f"{word_a} vs {word_b}",
                        "sharedTopics": sorted(common)[:5],
                    })

        # Also check reverse direction
        matches_a_rev = [s for s in sentences_a if word_b in s.lower()]
        matches_b_rev = [s for s in sentences_b if word_a in s.lower()]

        for sa in matches_a_rev:
            for sb in matches_b_rev:
                tokens_a = set(_tokenize(sa)) - {"the", "a", "is", "are", "was", "in", "of", "to"}
                tokens_b = set(_tokenize(sb)) - {"the", "a", "is", "are", "was", "in", "of", "to"}
                common = tokens_a & tokens_b - {word_a, word_b}
                if len(common) >= 2:
                    contradictions.append({
                        "statementA": sa,
                        "statementB": sb,
                        "oppositionPattern": f"{word_b} vs {word_a}",
                        "sharedTopics": sorted(common)[:5],
                    })

    # Deduplicate
    seen: set[str] = set()
    unique_contradictions: list[dict] = []
    for c in contradictions:
        key = f"{c['statementA'][:50]}|{c['statementB'][:50]}"
        if key not in seen:
            seen.add(key)
            unique_contradictions.append(c)

    return unique_contradictions[:10]
