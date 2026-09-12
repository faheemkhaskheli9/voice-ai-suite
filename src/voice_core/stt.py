"""
Whisper STT wrapper shared by every voice feature app (README.md Section
2/4, issue #2): one `SpeechToText` interface so feature apps call
`transcribe(AudioBuffer)` instead of each wiring the `whisper`/
`faster-whisper` library directly.

No GPU/real Whisper model download is available in this environment (or
CI), so `SpeechToText` talks to a pluggable `WhisperBackend` instead of
loading a real model directly — production code supplies a real backend
(e.g. one built on `openai-whisper` or `faster-whisper`); tests and this
module's default supply a fake one. That boundary is the mock this issue's
Definition of Done calls for, mirroring the `MicBackend` pattern in
`voice_core/audio.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .audio import AudioBuffer

#: Whisper's own published model sizes/versions -- kept here (rather than
#: accepting any string) so an unsupported size fails fast at construction
#: time instead of surfacing as an opaque backend error mid-transcription.
SUPPORTED_MODEL_SIZES = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
)


class TranscriptionError(ValueError):
    """Raised when a clip cannot be transcribed (backend failure, unusable
    audio) -- carries a clear `reason` instead of letting the caller see a
    raw backend exception or a crash."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class UnsupportedModelSize(TranscriptionError):
    """Raised when `SpeechToText` is configured with a model size Whisper
    doesn't publish."""

    def __init__(self, model_size: str):
        super().__init__(
            f"Unsupported Whisper model size {model_size!r}. "
            f"Supported: {', '.join(SUPPORTED_MODEL_SIZES)}."
        )


class NoSttBackendAvailable(TranscriptionError):
    """Raised by `SpeechToText` when no backend was supplied and no
    hardware/model-backed default is available in this environment (this is
    always true in CI and in this scaffold, per the Definition of Done: no
    real hardware or paid APIs)."""

    def __init__(self):
        super().__init__(
            "No Whisper backend available. Pass a WhisperBackend "
            "(a real one wrapping openai-whisper/faster-whisper for a real "
            "app, a fake one for tests)."
        )


@dataclass(frozen=True)
class TranscriptionResult:
    """One clip's transcription: the text plus enough metadata for feature
    apps to log/compare runs (e.g. the Phase 4 Whisper size/version
    comparison harness) without re-deriving it."""

    text: str
    model_size: str
    language: str | None = None
    segments: tuple[str, ...] = field(default_factory=tuple)


class WhisperBackend(Protocol):
    """What a real Whisper-model implementation must provide. Kept tiny and
    framework-free so it can be backed by `openai-whisper`,
    `faster-whisper`, a hosted API, or a test fake interchangeably."""

    def transcribe(
        self, samples: bytes, *, sample_rate: int, channels: int, model_size: str
    ) -> dict:
        """Transcribe raw PCM audio and return a dict with at least a
        ``text`` key; may also include ``language`` and ``segments``."""


class SpeechToText:
    """The one STT interface every feature app calls.

    Model size/version is a constructor argument (never hardcoded), so
    switching e.g. ``"base"`` -> ``"large-v3"`` is a config change, not a
    code change -- exactly what the Phase 4 size/version comparison harness
    needs to sweep over.
    """

    def __init__(self, backend: WhisperBackend | None = None, *, model_size: str = "base"):
        if model_size not in SUPPORTED_MODEL_SIZES:
            raise UnsupportedModelSize(model_size)
        self._backend = backend
        self.model_size = model_size

    def transcribe(self, audio: AudioBuffer) -> TranscriptionResult:
        if self._backend is None:
            raise NoSttBackendAvailable()
        if audio.duration_seconds <= 0:
            raise TranscriptionError("Cannot transcribe an empty/zero-duration clip.")

        try:
            raw = self._backend.transcribe(
                audio.samples,
                sample_rate=audio.sample_rate,
                channels=audio.channels,
                model_size=self.model_size,
            )
        except Exception as exc:  # noqa: BLE001 - any backend failure -> one clear error
            raise TranscriptionError(f"Whisper backend failed: {exc}") from exc

        text = raw.get("text")
        if not isinstance(text, str):
            raise TranscriptionError(
                "Whisper backend returned no usable 'text' field."
            )

        return TranscriptionResult(
            text=text,
            model_size=self.model_size,
            language=raw.get("language"),
            segments=tuple(raw.get("segments", ())),
        )
