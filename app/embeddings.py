"""TF-IDF based embedding system with SVD dimensionality reduction.

Provides dense vector embeddings without external API keys.
Falls back to hash-based approach when corpus is too small (< 2 documents).
"""

from __future__ import annotations

import hashlib
import threading
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


_EMBEDDING_DIM = 128
_MIN_CORPUS_SIZE = 2


class VectorizerStore:
    """Singleton that maintains a fitted TfidfVectorizer and SVD transformer."""

    _instance: Optional["VectorizerStore"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "VectorizerStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        self._svd: Optional[TruncatedSVD] = None
        self._corpus: list[str] = []
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def corpus_size(self) -> int:
        return len(self._corpus)

    def refit(self, corpus: list[str]) -> None:
        """Refit the vectorizer and SVD on the given corpus."""
        self._corpus = corpus
        if len(corpus) < _MIN_CORPUS_SIZE:
            self._fitted = False
            return

        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        tfidf_matrix = self._vectorizer.fit_transform(corpus)

        n_components = min(_EMBEDDING_DIM, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1])
        if n_components < 1:
            self._fitted = False
            return

        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf_matrix)
        self._fitted = True

    def transform(self, text: str) -> list[float]:
        """Transform a single text into a dense embedding vector."""
        if not self._fitted or self._svd is None:
            return _hash_embed(text, _EMBEDDING_DIM)

        tfidf_vec = self._vectorizer.transform([text])
        dense = self._svd.transform(tfidf_vec)[0]

        # Normalize to unit vector
        norm = np.linalg.norm(dense)
        if norm > 0:
            dense = dense / norm

        # Pad to full embedding dim if needed
        if len(dense) < _EMBEDDING_DIM:
            padded = np.zeros(_EMBEDDING_DIM)
            padded[: len(dense)] = dense
            dense = padded

        return dense.tolist()

    def reset(self) -> None:
        """Reset the store (useful for testing)."""
        self._corpus = []
        self._fitted = False
        self._svd = None
        self._vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (for testing only)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance._initialized = False
            cls._instance = None


def _hash_embed(text: str, dim: int = _EMBEDDING_DIM) -> list[float]:
    """Fallback hash-based embedding when corpus is too small."""
    vals = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        return vals
    for tok in tokens:
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        vals[h % dim] += 1.0
    scale = float(len(tokens))
    result = [v / scale for v in vals]
    # Normalize
    norm = sum(x * x for x in result) ** 0.5
    if norm > 0:
        result = [x / norm for x in result]
    return result


def embed(text: str) -> list[float]:
    """Embed text using the global VectorizerStore (TF-IDF + SVD or hash fallback)."""
    store = VectorizerStore()
    return store.transform(text)


def refit(corpus: list[str]) -> None:
    """Refit the global vectorizer on the given corpus."""
    store = VectorizerStore()
    store.refit(corpus)


def get_vectorizer_store() -> VectorizerStore:
    """Get the global VectorizerStore instance."""
    return VectorizerStore()
