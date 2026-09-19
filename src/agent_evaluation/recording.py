"""Recording backend for the Phase 3 evaluation pipeline (issue #11).

Turns a spoken line from a completed call (a persona utterance or an agent
response) into audio, so it can be transcribed and archived rather than
scored straight off the raw text the Phase 2 caller loop already carries.
No real microphone/telephony recording hardware is available in this
environment (or CI), so :class:`CallRecorder` talks to a pluggable
``RecordingBackend`` instead -- production code supplies a real one (e.g.
pulling the call's audio track off the LiveKit room); tests and this
module's default supply a fake one. Mirrors the `MicBackend`/`WhisperBackend`/
`TTSBackend` pattern in `voice_core`.
"""
from __future__ import annotations

from typing import Protocol

from voice_core.audio import AudioBuffer, AudioCaptureError


class NoRecordingBackendAvailable(AudioCaptureError):
    """Raised by `CallRecorder` when no backend was supplied and no
    hardware/network-backed default is available in this environment (this
    is always true in CI and in this scaffold, per the Definition of Done:
    no real hardware)."""

    def __init__(self):
        super().__init__(
            "No recording backend available. Pass a RecordingBackend "
            "(a real one capturing the call's audio track, a fake one for tests)."
        )


class RecordingBackend(Protocol):
    """What a real call-recording implementation must provide. Kept tiny
    and framework-free so it can be backed by a LiveKit track recorder, a
    telephony provider's call-recording API, or a test fake
    interchangeably."""

    def record(self, text: str, *, sample_rate: int, channels: int) -> bytes:
        """Return raw 16-bit PCM audio for the utterance `text`."""


class CallRecorder:
    """Records one spoken utterance at a time into an `AudioBuffer` via a
    pluggable `RecordingBackend`. With no backend supplied, `record_utterance`
    raises `NoRecordingBackendAvailable` rather than silently returning
    empty/fake audio -- a caller must explicitly wire a backend, real or
    fake."""

    def __init__(
        self,
        backend: RecordingBackend | None = None,
        *,
        sample_rate: int = 16_000,
        channels: int = 1,
    ):
        self._backend = backend
        self.sample_rate = sample_rate
        self.channels = channels

    def record_utterance(self, text: str) -> AudioBuffer:
        if self._backend is None:
            raise NoRecordingBackendAvailable()
        if not text.strip():
            raise AudioCaptureError("Cannot record an empty utterance.")

        samples = self._backend.record(
            text, sample_rate=self.sample_rate, channels=self.channels
        )
        return AudioBuffer(
            samples=samples,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width_bytes=2,
        )
