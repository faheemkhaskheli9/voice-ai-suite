"""Embedding provider interface (README.md Phase 5, issue #15) — swappable
per the knowledge-base "Chunking & embedding ingestion" pattern.

`HashingEmbedder` is the default: a deterministic, offline, CPU-only
embedder so ingestion, retrieval, and tests never need a paid API key or
network access (no paid APIs per the Definition of Done). A real semantic
provider (e.g. OpenAI/Cohere embeddings) is a config-time swap-in that
implements the same `EmbeddingProvider` interface — not exercised here since
this suite is CPU-only.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dimensions(self) -> int:
        raise NotImplementedError


class HashingEmbedder(EmbeddingProvider):
    """Deterministic bag-of-words hashing embedder: each token is hashed
    into one of `dimensions` buckets and counted, then L2-normalized. Not a
    semantic embedding, but sufficient to prove a working ingestion +
    retrieval pipeline without any model download or API call.
    """

    def __init__(self, dimensions: int = 256):
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimensions, dtype=np.float64)
        for token in text.lower().split():
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self._dimensions
            vector[bucket] += 1.0
        norm = math.sqrt(float(np.dot(vector, vector)))
        if norm > 0:
            vector /= norm
        return vector

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self._embed_one(t) for t in texts]
