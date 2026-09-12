"""
Audio input abstraction shared by every voice feature app (README.md
Section 2/4): mic capture and file capture both produce the same
`AudioBuffer`, so downstream Whisper STT wrapping (issue #2) never needs to
know which source a clip came from.

No real microphone hardware is available in this environment (or CI), so
`MicAudioSource` talks to a pluggable `MicBackend` instead of a hardware
library directly — production code supplies a real backend (e.g. one built
on `sounddevice`/`pyaudio`); tests and this module's default supply a fake
one. That boundary is the mock this issue's Definition of Done calls for.
"""
from __future__ import annotations

import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol


class AudioCaptureError(ValueError):
    """Raised when an audio source cannot produce audio (bad file, backend
    failure, unsupported format) — carries a clear `reason`."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class AudioBuffer:
    """The one internal audio representation every source (mic or file)
    yields, and every downstream consumer (STT) reads.

    `samples` is raw little-endian PCM (16-bit signed, mono or interleaved
    multi-channel per `channels`) — the shape `wave` and every common STT
    library agree on, so no source-specific decoding leaks downstream.
    """

    samples: bytes
    sample_rate: int
    channels: int
    sample_width_bytes: int = 2  # 16-bit PCM

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate == 0:
            return 0.0
        frame_size = self.channels * self.sample_width_bytes
        n_frames = len(self.samples) // frame_size if frame_size else 0
        return n_frames / self.sample_rate


class AudioSource(ABC):
    """Common interface for anything that can produce an `AudioBuffer`."""

    @abstractmethod
    def capture(self) -> AudioBuffer:
        """Capture/read the full clip and return it as one `AudioBuffer`."""


class FileAudioSource(AudioSource):
    """Reads a local WAV file into an `AudioBuffer`.

    Uses the stdlib `wave` module rather than a third-party decoder so
    Phase 1 has zero extra runtime dependencies; a later phase can add
    additional-format support (mp3/flac) behind the same interface without
    changing any caller.
    """

    def __init__(self, path: str | Path | BinaryIO):
        self._path = path

    def capture(self) -> AudioBuffer:
        # `wave.open` accepts a str path or an already-open binary file, but
        # not a bare Path object on every Python version — normalize here.
        path = str(self._path) if isinstance(self._path, Path) else self._path
        try:
            with wave.open(path, "rb") as wf:  # type: ignore[arg-type]
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                samples = wf.readframes(wf.getnframes())
        except (wave.Error, EOFError, OSError) as exc:
            raise AudioCaptureError(f"Could not read audio file: {exc}") from exc

        if sample_rate <= 0:
            raise AudioCaptureError("Audio file reports an invalid sample rate.")

        return AudioBuffer(
            samples=samples,
            sample_rate=sample_rate,
            channels=n_channels,
            sample_width_bytes=sample_width,
        )


class MicBackend(Protocol):
    """What a real microphone-capture implementation must provide. Kept
    tiny and framework-free so it can be backed by `sounddevice`,
    `pyaudio`, a LiveKit audio track, or a test fake interchangeably."""

    def record(self, duration_seconds: float, sample_rate: int, channels: int) -> bytes:
        """Return `duration_seconds` of raw 16-bit PCM audio."""


class NoMicBackendAvailable(AudioCaptureError):
    """Raised by `MicAudioSource` when no backend was supplied and no
    hardware-backed default is available in this environment (this is
    always true in CI and in this scaffold, per the Definition of Done: no
    real hardware)."""

    def __init__(self):
        super().__init__(
            "No microphone backend available. Pass a MicBackend "
            "(a real one for a desktop app, a fake one for tests)."
        )


class MicAudioSource(AudioSource):
    """Captures a fixed-duration clip from a microphone via a pluggable
    `MicBackend`. With no backend supplied, capture() raises
    `NoMicBackendAvailable` rather than silently returning empty/fake audio
    — a caller must explicitly wire a backend, real or fake."""

    def __init__(
        self,
        backend: MicBackend | None = None,
        *,
        duration_seconds: float = 3.0,
        sample_rate: int = 16_000,
        channels: int = 1,
    ):
        self._backend = backend
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels

    def capture(self) -> AudioBuffer:
        if self._backend is None:
            raise NoMicBackendAvailable()

        samples = self._backend.record(
            self.duration_seconds, self.sample_rate, self.channels
        )
        return AudioBuffer(
            samples=samples,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width_bytes=2,
        )
