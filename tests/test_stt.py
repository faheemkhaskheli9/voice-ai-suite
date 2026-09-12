import pytest

from voice_core.audio import AudioBuffer
from voice_core.stt import (
    NoSttBackendAvailable,
    SpeechToText,
    TranscriptionError,
    UnsupportedModelSize,
)


class FakeWhisperBackend:
    """Test double standing in for a real Whisper model -- see
    voice_core/stt.py's module docstring for why no real model/GPU is
    available in this environment."""

    def __init__(self, text: str = "hello world", language: str = "en", fail: bool = False):
        self.text = text
        self.language = language
        self.fail = fail
        self.last_call = None

    def transcribe(self, samples: bytes, *, sample_rate: int, channels: int, model_size: str) -> dict:
        self.last_call = (len(samples), sample_rate, channels, model_size)
        if self.fail:
            raise RuntimeError("model crashed")
        return {"text": self.text, "language": self.language, "segments": [self.text]}


def _clip(n_frames=8_000, sample_rate=16_000, channels=1) -> AudioBuffer:
    return AudioBuffer(
        samples=b"\x00\x01" * n_frames * channels,
        sample_rate=sample_rate,
        channels=channels,
    )


def test_transcribe_returns_text_and_metadata():
    backend = FakeWhisperBackend(text="turn on the lights")
    stt = SpeechToText(backend, model_size="small")

    result = stt.transcribe(_clip())

    assert result.text == "turn on the lights"
    assert result.model_size == "small"
    assert result.language == "en"
    assert result.segments == ("turn on the lights",)
    assert backend.last_call == (8_000 * 2, 16_000, 1, "small")


def test_model_size_is_configurable_not_hardcoded():
    backend = FakeWhisperBackend()
    for size in ("tiny", "base", "large-v3"):
        stt = SpeechToText(backend, model_size=size)
        result = stt.transcribe(_clip())
        assert result.model_size == size


def test_unsupported_model_size_rejected_at_construction():
    with pytest.raises(UnsupportedModelSize):
        SpeechToText(FakeWhisperBackend(), model_size="xl-turbo")


def test_no_backend_raises_clear_error_instead_of_crashing():
    stt = SpeechToText(model_size="base")
    with pytest.raises(NoSttBackendAvailable):
        stt.transcribe(_clip())


def test_backend_failure_raises_transcription_error_not_the_raw_exception():
    stt = SpeechToText(FakeWhisperBackend(fail=True))
    with pytest.raises(TranscriptionError, match="Whisper backend failed"):
        stt.transcribe(_clip())


def test_empty_clip_raises_clear_error():
    stt = SpeechToText(FakeWhisperBackend())
    empty = AudioBuffer(samples=b"", sample_rate=16_000, channels=1)
    with pytest.raises(TranscriptionError, match="empty"):
        stt.transcribe(empty)


def test_backend_returning_no_text_raises_clear_error():
    class BadBackend:
        def transcribe(self, samples, *, sample_rate, channels, model_size):
            return {"language": "en"}  # no 'text' key

    stt = SpeechToText(BadBackend())
    with pytest.raises(TranscriptionError, match="no usable"):
        stt.transcribe(_clip())
