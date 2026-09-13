import pytest

from voice_core.tts import (
    NoTtsBackendAvailable,
    SynthesisError,
    TextToSpeech,
    UnsupportedProvider,
)


class FakeTTSBackend:
    """Test double standing in for a real TTS provider -- see
    voice_core/tts.py's module docstring for why no real engine/network
    call is available in this environment."""

    def __init__(self, audio: bytes = b"fake-audio-bytes", fail: bool = False):
        self.audio = audio
        self.fail = fail
        self.last_call = None

    def synthesize(self, text: str, *, voice: str | None) -> bytes:
        self.last_call = (text, voice)
        if self.fail:
            raise RuntimeError("provider rejected request")
        return self.audio


def test_synthesize_writes_audio_file_and_returns_metadata(tmp_path):
    backend = FakeTTSBackend(audio=b"some audio")
    tts = TextToSpeech(backend, provider="pyttsx3", voice="default")
    out = tmp_path / "out.mp3"

    result = tts.synthesize("hello there", out)

    assert result.audio_path == str(out)
    assert result.provider == "pyttsx3"
    assert result.voice == "default"
    assert out.read_bytes() == b"some audio"
    assert backend.last_call == ("hello there", "default")


def test_provider_is_configurable_not_hardcoded(tmp_path):
    backend = FakeTTSBackend()
    for provider in ("pyttsx3", "gtts", "elevenlabs"):
        tts = TextToSpeech(backend, provider=provider)
        result = tts.synthesize("configurable provider", tmp_path / f"out_{provider}.mp3")
        assert result.provider == provider


def test_voice_selection_is_configuration_not_a_code_change(tmp_path):
    backend = FakeTTSBackend()
    for voice in ("voice-a", "voice-b"):
        tts = TextToSpeech(backend, voice=voice)
        tts.synthesize("hi", tmp_path / f"out_{voice}.mp3")
        assert backend.last_call[1] == voice


def test_unsupported_provider_rejected_at_construction():
    with pytest.raises(UnsupportedProvider):
        TextToSpeech(FakeTTSBackend(), provider="not-a-real-provider")


def test_no_backend_raises_clear_error_instead_of_crashing(tmp_path):
    tts = TextToSpeech(provider="pyttsx3")
    with pytest.raises(NoTtsBackendAvailable):
        tts.synthesize("hi", tmp_path / "out.mp3")


def test_backend_failure_raises_synthesis_error_not_the_raw_exception(tmp_path):
    tts = TextToSpeech(FakeTTSBackend(fail=True))
    with pytest.raises(SynthesisError, match="TTS backend failed"):
        tts.synthesize("hi", tmp_path / "out.mp3")


def test_empty_text_raises_clear_error():
    tts = TextToSpeech(FakeTTSBackend())
    with pytest.raises(SynthesisError, match="empty"):
        tts.synthesize("   ", "unused.mp3")


def test_backend_returning_no_audio_raises_clear_error(tmp_path):
    class SilentBackend:
        def synthesize(self, text, *, voice):
            return b""

    tts = TextToSpeech(SilentBackend())
    with pytest.raises(SynthesisError, match="no audio bytes"):
        tts.synthesize("hi", tmp_path / "out.mp3")
