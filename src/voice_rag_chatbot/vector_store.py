"""Vector store interface (README.md Phase 5, issue #15) — swappable per the
knowledge-base "Chunking & embedding ingestion" pattern. A real deployment
would point this at Chroma/pgvector/Pinecone; `JSONVectorStore` is a
dependency-free local implementation so this phase is runnable without
standing up external infrastructure.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VectorRecord:
    id: str
    embedding: list[float]
    document: str
    metadata: dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: list[VectorRecord]) -> None:
        """Insert or replace records by id — re-ingesting the same id must
        not create a duplicate (idempotent re-ingestion, per the KB
        pattern)."""
        raise NotImplementedError

    @abstractmethod
    def query_with_scores(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple]:
        """Top-`top_k` records by cosine similarity to `embedding`, each
        paired with its similarity score so a caller can apply a threshold."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError


class JSONVectorStore(VectorStore):
    """Persists records as one JSON file, written atomically (portfolio
    rule 1) so an interrupted ingestion run never leaves a truncated store.
    Fine for a portfolio-scale demo corpus; not intended for production-scale
    corpora.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: dict[str, VectorRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        for item in raw:
            record = VectorRecord(**item)
            self._records[record.id] = record

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            [
                {"id": r.id, "embedding": r.embedding, "document": r.document, "metadata": r.metadata}
                for r in self._records.values()
            ]
        ).encode("utf-8")

        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
            raise

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self._records[record.id] = record  # upsert by id -> idempotent
        self._save()

    def query_with_scores(self, embedding: np.ndarray, top_k: int = 5) -> list[tuple]:
        if not self._records:
            return []
        scored = []
        query_vec = np.asarray(embedding, dtype=np.float64)
        for record in self._records.values():
            vec = np.asarray(record.embedding, dtype=np.float64)
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(vec)) or 1.0
            score = float(np.dot(query_vec, vec) / denom)
            scored.append((score, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._records)
