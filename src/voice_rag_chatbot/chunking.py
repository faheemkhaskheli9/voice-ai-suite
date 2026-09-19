"""Fixed-size character chunking with overlap (README.md Phase 5, issue #15).

Applies the knowledge-base "Chunking & embedding ingestion" pattern
(E:\\Projects\\LLM\\knowledge-base\\rag\\chunking-and-embedding-ingestion.md):
boundaries snap to the nearest preceding whitespace so words aren't split
mid-token, and overlap is guarded to always make forward progress.
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """Splits `text` into overlapping chunks of at most `chunk_size` chars."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        next_start = end - chunk_overlap
        # Guard against the whitespace-snapped `end` landing so close to
        # `start` that overlap would produce zero forward progress.
        start = next_start if next_start > start else end
    return chunks
