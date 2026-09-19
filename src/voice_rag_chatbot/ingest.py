"""Document ingestion pipeline (README.md Phase 5, issue #15): reads text
documents from a directory, chunks them, embeds each chunk, and upserts
into the configured vector store with source metadata for later citation --
the knowledge-base "Chunking & embedding ingestion" pattern.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from .chunking import chunk_text
from .embeddings import EmbeddingProvider
from .vector_store import VectorRecord, VectorStore

DEFAULT_EXTENSIONS = {".txt", ".md"}


class IngestionError(ValueError):
    pass


def _chunk_id(source: str, chunk_index: int, chunk: str) -> str:
    """Deterministic id from source path + chunk index + chunk content, so
    re-ingesting the same source produces the same ids (idempotent upsert)
    but an edited chunk gets a new id rather than silently overwriting the
    old text under a stale key."""
    return hashlib.sha256(f"{source}::{chunk_index}::{chunk}".encode("utf-8")).hexdigest()


def iter_source_files(
    source_dir: str | Path, extensions: Iterable[str] = DEFAULT_EXTENSIONS
) -> list[Path]:
    source_root = Path(source_dir)
    if not source_root.is_dir():
        raise IngestionError(f"source directory not found: {source_root}")
    exts = {e.lower() for e in extensions}
    return sorted(p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


class IngestionPipeline:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: str | Path) -> int:
        """Ingests one file; returns the number of chunks written."""
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0

        embeddings = self.embedder.embed(chunks)
        records = [
            VectorRecord(
                id=_chunk_id(str(path), i, chunk),
                embedding=embedding.tolist(),
                document=chunk,
                metadata={"source": str(path), "title": path.stem, "chunk_index": i},
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self.store.upsert(records)
        return len(records)

    def ingest_directory(self, source_dir: str | Path) -> int:
        total = 0
        for path in iter_source_files(source_dir):
            total += self.ingest_file(path)
        return total
