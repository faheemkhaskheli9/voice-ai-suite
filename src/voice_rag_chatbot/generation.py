"""LLM response generation grounded in retrieved context (README.md Phase
5, issue #16).

No real LLM API is available in this environment (or CI) -- no paid APIs
per the Definition of Done -- so generation goes through a pluggable
`LLMBackend` instead of an LLM API directly, mirroring the
`WhisperBackend`/`TTSBackend` pattern in `voice_core.stt`/`voice_core.tts`
and the `Planner` pattern in `realtime_agent.dialogue`. Production code
supplies a real backend; tests supply a fake one.

A retrieval miss (`RetrievalResult.chunks` empty) never reaches the LLM
backend at all -- it returns a fixed, clearly-labeled generic response
instead. Prompting an LLM with no context and trusting it to say "I don't
know" risks a fabricated, uncited answer; short-circuiting before the call
is the only way to guarantee this issue's "not a fabricated citation"
acceptance criterion holds for every backend, not just well-behaved ones.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .retrieval import RetrievalResult

#: Fixed fallback text for a retrieval miss -- deliberately generic and
#: clearly labeled so it's never mistaken for a grounded, sourced answer.
NO_CONTEXT_RESPONSE = (
    "I don't have any information about that in my knowledge base yet, "
    "so I can't give you a sourced answer."
)


class LLMBackend(Protocol):
    """What a real LLM implementation must provide. Kept tiny and
    framework-free so it can be backed by any provider, or a test fake."""

    def generate(self, prompt: str) -> str:
        """Return a response for `prompt` (a full grounded prompt already
        containing the retrieved context and the user's query)."""


@dataclass(frozen=True)
class SourceCitation:
    source: str
    document: str
    score: float


@dataclass(frozen=True)
class GroundedResponse:
    text: str
    grounded: bool
    sources: tuple[SourceCitation, ...]


class GenerationError(ValueError):
    pass


def _build_prompt(query: str, result: RetrievalResult) -> str:
    context_block = "\n\n".join(
        f"[{i}] (source: {chunk.metadata.get('source', 'unknown')}) {chunk.document}"
        for i, chunk in enumerate(result.chunks, start=1)
    )
    return (
        "Answer the user's question using only the numbered context below. "
        "Cite the context number(s) you used.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )


def _format_sources_line(sources: tuple[SourceCitation, ...]) -> str:
    names = ", ".join(dict.fromkeys(c.source for c in sources))
    return f"\n\nSources: {names}"


class ResponseGenerator:
    """Generates an LLM response conditioned on `RagRetriever`'s output."""

    def __init__(self, backend: LLMBackend):
        self._backend = backend

    def generate(self, result: RetrievalResult) -> GroundedResponse:
        if not result.chunks:
            return GroundedResponse(text=NO_CONTEXT_RESPONSE, grounded=False, sources=())

        prompt = _build_prompt(result.query_text, result)
        try:
            raw_text = self._backend.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - any backend failure -> one clear error
            raise GenerationError(f"LLM backend failed: {exc}") from exc

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise GenerationError("LLM backend returned no usable text.")

        sources = tuple(
            SourceCitation(
                source=chunk.metadata.get("source", "unknown"),
                document=chunk.document,
                score=chunk.score,
            )
            for chunk in result.chunks
        )
        text = raw_text.strip() + _format_sources_line(sources)
        return GroundedResponse(text=text, grounded=True, sources=sources)
