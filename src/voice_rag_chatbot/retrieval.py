"""Mic/file capture -> transcription -> RAG retrieval (README.md Phase 5,
issue #15).

Reuses the Phase 1 shared `voice_core.audio.AudioSource` abstraction (mic
or file, same `AudioBuffer` either way) and `voice_core.stt.SpeechToText`
for transcription, then embeds the transcribed query and retrieves the
top-k most relevant chunks from a `VectorStore` populated by
`ingest.IngestionPipeline` -- the retrieval half of the knowledge-base
"Chunking & embedding ingestion" pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

from voice_core.audio import AudioSource
from voice_core.stt import SpeechToText

from .embeddings import EmbeddingProvider
from .vector_store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    document: str
    score: float
    metadata: dict


@dataclass(frozen=True)
class RetrievalResult:
    query_text: str
    chunks: tuple[RetrievedChunk, ...]


class RagRetriever:
    """Embeds a query (spoken or typed) and retrieves the top-k most
    similar chunks from the configured vector store, filtered by a minimum
    similarity score."""

    def __init__(
        self,
        stt: SpeechToText,
        embedder: EmbeddingProvider,
        store: VectorStore,
        *,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between -1.0 and 1.0")
        self._stt = stt
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve_from_audio(self, audio_source: AudioSource) -> RetrievalResult:
        """Captures audio via `audio_source` (mic or file -- both yield the
        same `AudioBuffer`), transcribes it, and retrieves matching
        context. A capture or transcription failure propagates rather than
        being silently treated as an empty/no-match query."""
        audio = audio_source.capture()
        transcription = self._stt.transcribe(audio)
        return self.retrieve_from_text(transcription.text)

    def retrieve_from_text(self, query: str) -> RetrievalResult:
        """Retrieves matching context for an already-transcribed (or typed)
        query. Never raises for an empty query or an empty store -- both
        simply produce no matches."""
        query = (query or "").strip()
        if not query or len(self._store) == 0:
            return RetrievalResult(query_text=query, chunks=())

        [embedding] = self._embedder.embed([query])
        scored = self._store.query_with_scores(embedding, top_k=self._top_k)
        chunks = tuple(
            RetrievedChunk(document=record.document, score=score, metadata=dict(record.metadata))
            for score, record in scored
            if score >= self._score_threshold
        )
        return RetrievalResult(query_text=query, chunks=chunks)
