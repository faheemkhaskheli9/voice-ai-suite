import pytest

from agent_evaluation.recording import CallRecorder, NoRecordingBackendAvailable
from voice_core.audio import AudioCaptureError


class FakeRecordingBackend:
    """Test double standing in for a real call-recording implementation --
    see agent_evaluation/recording.py's module docstring for why no real
    hardware/network recorder is available in this environment."""

    def __init__(self):
        self.last_call = None

    def record(self, text: str, *, sample_rate: int, channels: int) -> bytes:
        self.last_call = (text, sample_rate, channels)
        # one frame per character keeps the buffer's duration proportional
        # to the utterance, like a real recording would be.
        return b"\x00\x01" * len(text) * channels


def test_records_utterance_into_an_audio_buffer():
    backend = FakeRecordingBackend()
    recorder = CallRecorder(backend, sample_rate=16_000, channels=1)

    audio = recorder.record_utterance("hello there")

    assert audio.sample_rate == 16_000
    assert audio.channels == 1
    assert audio.duration_seconds > 0
    assert backend.last_call == ("hello there", 16_000, 1)


def test_sample_rate_and_channels_are_configurable_not_hardcoded():
    backend = FakeRecordingBackend()
    recorder = CallRecorder(backend, sample_rate=8_000, channels=2)

    audio = recorder.record_utterance("hi")

    assert audio.sample_rate == 8_000
    assert audio.channels == 2


def test_no_backend_raises_clear_error_instead_of_crashing():
    recorder = CallRecorder()
    with pytest.raises(NoRecordingBackendAvailable):
        recorder.record_utterance("hello")


def test_empty_utterance_raises_clear_error():
    recorder = CallRecorder(FakeRecordingBackend())
    with pytest.raises(AudioCaptureError, match="empty"):
        recorder.record_utterance("   ")
