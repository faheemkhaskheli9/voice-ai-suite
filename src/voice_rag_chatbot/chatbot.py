"""End-to-end Voice RAG Chatbot loop: capture -> transcribe -> retrieve ->
generate -> speak (README.md Phase 5, issue #17).

Reuses the Phase 1 `voice_core.tts.TextToSpeech` wrapper for the speak step,
same as every other feature app in this suite. A TTS failure is caught and
turned into `audio_path=None` rather than propagating -- the generated,
sourced text answer (`generation.py`'s whole point) must not be lost just
because the last, purely cosmetic step (speaking it aloud) failed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voice_core.audio import AudioSource
from voice_core.tts import SynthesisError, TextToSpeech

from .generation import GroundedResponse, ResponseGenerator
from .retrieval import RagRetriever


@dataclass(frozen=True)
class ChatbotTurnResult:
    response: GroundedResponse
    audio_path: str | None
    """Path to the spoken answer, or `None` if TTS synthesis failed --
    `response` is always populated regardless."""


class VoiceRagChatbot:
    """Wires `RagRetriever` -> `ResponseGenerator` -> `TextToSpeech` into
    one turn: a voice or text query in, a grounded spoken (or at least
    written) answer out."""

    def __init__(self, retriever: RagRetriever, generator: ResponseGenerator, tts: TextToSpeech):
        self._retriever = retriever
        self._generator = generator
        self._tts = tts

    def handle_audio_query(
        self, audio_source: AudioSource, output_path: str | Path
    ) -> ChatbotTurnResult:
        """Full loop: capture -> transcribe -> retrieve -> generate -> speak."""
        retrieval = self._retriever.retrieve_from_audio(audio_source)
        return self._respond(retrieval.query_text, retrieval, output_path)

    def handle_text_query(self, query: str, output_path: str | Path) -> ChatbotTurnResult:
        """Same loop, skipping audio capture/transcription for an
        already-typed query."""
        retrieval = self._retriever.retrieve_from_text(query)
        return self._respond(query, retrieval, output_path)

    def _respond(self, query, retrieval, output_path) -> ChatbotTurnResult:
        response = self._generator.generate(retrieval)
        audio_path = self._speak(response.text, output_path)
        return ChatbotTurnResult(response=response, audio_path=audio_path)

    def _speak(self, text: str, output_path: str | Path) -> str | None:
        try:
            result = self._tts.synthesize(text, output_path)
        except SynthesisError:
            return None
        return result.audio_path
