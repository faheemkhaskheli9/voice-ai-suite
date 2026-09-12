import struct
import wave

import pytest

from voice_core.audio import (
    AudioBuffer,
    AudioCaptureError,
    FileAudioSource,
    MicAudioSource,
    NoMicBackendAvailable,
)


class FakeMicBackend:
    """Test double standing in for real microphone hardware — see
    voice_core/audio.py's module docstring for why no real backend is
    available in this environment."""

    def __init__(self, tone_amplitude: int = 1000):
        self.tone_amplitude = tone_amplitude
        self.last_call = None

    def record(self, duration_seconds: float, sample_rate: int, channels: int) -> bytes:
        self.last_call = (duration_seconds, sample_rate, channels)
        n_frames = int(duration_seconds * sample_rate)
        return struct.pack(f"<{n_frames * channels}h", *([self.tone_amplitude] * n_frames * channels))


def _write_wav(path, *, sample_rate=16_000, channels=1, sample_width=2, n_frames=8_000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_frames * channels}h", *([500] * n_frames * channels)))


def test_file_audio_source_reads_wav(tmp_path):
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, sample_rate=16_000, channels=1, n_frames=8_000)

    buf = FileAudioSource(wav_path).capture()

    assert isinstance(buf, AudioBuffer)
    assert buf.sample_rate == 16_000
    assert buf.channels == 1
    assert buf.sample_width_bytes == 2
    assert len(buf.samples) == 8_000 * 2
    assert buf.duration_seconds == pytest.approx(0.5)


def test_file_audio_source_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(AudioCaptureError):
        FileAudioSource(tmp_path / "does-not-exist.wav").capture()


def test_file_audio_source_rejects_non_wav_bytes(tmp_path):
    bad = tmp_path / "not-a-wav.wav"
    bad.write_bytes(b"this is not a wav file")

    with pytest.raises(AudioCaptureError, match="Could not read audio file"):
        FileAudioSource(bad).capture()


def test_mic_audio_source_without_backend_raises():
    with pytest.raises(NoMicBackendAvailable):
        MicAudioSource().capture()


def test_mic_audio_source_uses_fake_backend():
    backend = FakeMicBackend()
    source = MicAudioSource(backend, duration_seconds=1.0, sample_rate=16_000, channels=1)

    buf = source.capture()

    assert isinstance(buf, AudioBuffer)
    assert buf.sample_rate == 16_000
    assert buf.channels == 1
    assert backend.last_call == (1.0, 16_000, 1)
    assert buf.duration_seconds == pytest.approx(1.0)


def test_mic_and_file_sources_yield_the_same_internal_representation(tmp_path):
    """Core acceptance criterion: downstream STT must not care which
    source produced the AudioBuffer."""
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, sample_rate=16_000, channels=1, n_frames=16_000)
    file_buf = FileAudioSource(wav_path).capture()

    mic_buf = MicAudioSource(
        FakeMicBackend(), duration_seconds=1.0, sample_rate=16_000, channels=1
    ).capture()

    assert type(file_buf) is type(mic_buf) is AudioBuffer
    assert file_buf.sample_rate == mic_buf.sample_rate
    assert file_buf.channels == mic_buf.channels
    assert file_buf.sample_width_bytes == mic_buf.sample_width_bytes
