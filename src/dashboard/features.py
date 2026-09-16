"""The 4 feature-picker entries shown on the dashboard.

A plain list for now — this scaffold issue only needs a shell. Once each
feature app lands (Phases 2-5), this becomes the seed for the registry
pattern docs/architecture.md describes, rather than a shape that needs
reworking.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    slug: str
    title: str
    description: str
    url_name: str | None = None
    """Name of the real feature app's own URL to redirect to once it exists
    (Phases 2-5). None means it isn't built yet, so the dashboard link falls
    back to `feature_stub`'s placeholder page."""


FEATURES: tuple[Feature, ...] = (
    Feature(
        slug="realtime-voice-agent",
        title="Real-Time Voice Agent",
        description=(
            "LiveKit real-time audio pipeline with streaming STT/TTS, "
            "barge-in handling, conversation memory, and tool calling."
        ),
    ),
    Feature(
        slug="voice-agent-evaluation",
        title="Voice Agent Evaluation",
        description=(
            "Configurable test personas run simulated conversations against "
            "the real-time agent with LLM-as-judge scoring and pass/fail "
            "flagging."
        ),
    ),
    Feature(
        slug="speech-model-evaluator",
        title="Speech Model Evaluator",
        description=(
            "Whisper size/version comparison, TTS provider/voice comparison, "
            "WER measurement, latency benchmarking, and cost tracking."
        ),
    ),
    Feature(
        slug="voice-rag-chatbot",
        title="Voice RAG Chatbot",
        description=(
            "Microphone/file input capture, speech recognition, RAG-based "
            "retrieval, LLM response generation, and text-to-speech output."
        ),
    ),
)

FEATURES_BY_SLUG: dict[str, Feature] = {f.slug: f for f in FEATURES}
