"""
Pluggable TTS wrapper shared by every voice feature app (README.md Section
2/4, issue #3): one `TextToSpeech` interface so feature apps call
`synthesize(text, output_path)` instead of each wiring pyttsx3/gTTS/a cloud
provider directly -- and swapping provider or voice is a configuration
change, never a code change in a caller.

No real TTS engine or network-backed cloud provider is available in this
environment (or CI), so `TextToSpeech` talks to a pluggable `TTSBackend`
instead of calling pyttsx3/gTTS/a cloud SDK directly -- production code
supplies a real backend per provider (e.g. one built on `pyttsx3`, `gTTS`,
or a cloud provider such as ElevenLabs per README Section 3 Tech Stack);
tests and this module's default supply a fake one. That boundary is the
mock this issue's Definition of Done calls for, mirroring the
`MicBackend`/`WhisperBackend` pattern in `voice_core/audio.py`/
`voice_core/stt.py`. Ported from `python-voice-rag-chatbot.synthesis`'s
pyttsx3/gTTS engine choice, generalized to a third pluggable provider slot
for a real-time cloud provider.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

#: Providers this wrapper knows how to select among -- kept here (rather
#: than accepting any string) so an unsupported provider fails fast at
#: construction time instead of surfacing as an opaque backend error mid-
#: synthesis. Mirrors `stt.SUPPORTED_MODEL_SIZES`.
SUPPORTED_PROVIDERS = ("pyttsx3", "gtts", "elevenlabs")


class SynthesisError(ValueError):
    """Raised when text cannot be synthesized (backend failure, unusable
    input) -- carries a clear `reason` instead of letting the caller see a
    raw backend exception or a crash."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class UnsupportedProvider(SynthesisError):
    """Raised when `TextToSpeech` is configured with a provider this
    wrapper doesn't know about."""

    def __init__(self, provider: str):
        super().__init__(
            f"Unsupported TTS provider {provider!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )


class NoTtsBackendAvailable(SynthesisError):
    """Raised by `TextToSpeech` when no backend was supplied and no
    hardware/network-backed default is available in this environment (this
    is always true in CI and in this scaffold, per the Definition of Done:
    no real hardware or paid APIs)."""

    def __init__(self):
        super().__init__(
            "No TTS backend available. Pass a TTSBackend "
            "(a real one wrapping pyttsx3/gTTS/a cloud provider for a real "
            "app, a fake one for tests)."
        )


@dataclass(frozen=True)
class SynthesisResult:
    """One synthesis attempt's outcome: where the audio landed plus enough
    metadata for feature apps to log/compare runs without re-deriving it."""

    audio_path: str
    provider: str
    voice: str | None = None


class TTSBackend(Protocol):
    """What a real TTS-provider implementation must provide. Kept tiny and
    framework-free so it can be backed by `pyttsx3`, `gTTS`, a cloud
    provider (e.g. ElevenLabs), or a test fake interchangeably."""

    def synthesize(self, text: str, *, voice: str | None) -> bytes:
        """Return raw audio bytes for `text` spoken in `voice` (encoding is
        provider-specific; the caller only ever writes the bytes to a
        file, never inspects them)."""


class TextToSpeech:
    """The one TTS interface every feature app calls.

    Provider/voice is a constructor argument (never hardcoded), so
    swapping e.g. ``"pyttsx3"`` -> ``"elevenlabs"`` -- or just picking a
    different voice on the same provider -- is a config change, not a code
    change in any caller (this issue's acceptance criteria).
    """

    def __init__(
        self,
        backend: TTSBackend | None = None,
        *,
        provider: str = "pyttsx3",
        voice: str | None = None,
    ):
        if provider not in SUPPORTED_PROVIDERS:
            raise UnsupportedProvider(provider)
        self._backend = backend
        self.provider = provider
        self.voice = voice

    def synthesize(self, text: str, output_path: str | Path) -> SynthesisResult:
        if self._backend is None:
            raise NoTtsBackendAvailable()
        if not text.strip():
            raise SynthesisError("Cannot synthesize empty text.")

        try:
            audio_bytes = self._backend.synthesize(text, voice=self.voice)
        except Exception as exc:  # noqa: BLE001 - any backend failure -> one clear error
            raise SynthesisError(f"TTS backend failed: {exc}") from exc

        if not audio_bytes:
            raise SynthesisError(
                "TTS backend reported success but produced no audio bytes."
            )

        path = Path(output_path)
        path.write_bytes(audio_bytes)
        return SynthesisResult(audio_path=str(path), provider=self.provider, voice=self.voice)
