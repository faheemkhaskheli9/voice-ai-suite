import io
import struct
import wave
from pathlib import Path

import pytest

from speech_evaluator.tts_comparison import (
    TtsComparisonError,
    TtsVariant,
    load_tts_comparison_results,
    run_tts_comparison,
    write_tts_comparison_results,
)
from voice_core.stt import TranscriptionError
from voice_core.tts import SynthesisError

SAMPLE_RATE = 8000


def _wav_bytes(seconds, rate=SAMPLE_RATE):
    n = int(seconds * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * n, *([0] * n)))
    return buf.getvalue()


def _pcm_len(seconds, rate=SAMPLE_RATE):
    return int(seconds * rate) * 2


TEXTS = ["hello world", "the quick brown fox"]


class _FakeTTSBackend:
    """Returns a WAV clip whose length is keyed by reference text, so a
    fake STT backend downstream can tell which text/variant produced a
    clip -- standing in for a real provider whose audio actually differs
    in quality per (provider, voice)."""

    def __init__(self, duration_by_text: dict[str, float]):
        self._durations = duration_by_text

    def synthesize(self, text: str, *, voice: str | None) -> bytes:
        return _wav_bytes(self._durations[text])


class _PerClipSttBackend:
    """Returns a canned transcription keyed by (model_size, clip length),
    mirroring the fake used for the Whisper size/version comparison harness
    (test_speech_evaluator_comparison.py)."""

    def __init__(self, responses: dict[tuple[str, int], str]):
        self._responses = responses

    def transcribe(self, samples: bytes, *, sample_rate: int, channels: int, model_size: str) -> dict:
        return {"text": self._responses[(model_size, len(samples))]}


@pytest.fixture
def good_variant():
    durations = {"hello world": 0.5, "the quick brown fox": 1.0}
    return TtsVariant(provider="pyttsx3", voice="default", backend=_FakeTTSBackend(durations)), durations


@pytest.fixture
def bad_variant():
    durations = {"hello world": 0.7, "the quick brown fox": 1.3}
    return TtsVariant(provider="gtts", voice="alt", backend=_FakeTTSBackend(durations)), durations


def test_compares_multiple_providers_and_voices(tmp_path, good_variant, bad_variant):
    good, good_durations = good_variant
    bad, bad_durations = bad_variant
    stt_backend = _PerClipSttBackend(
        {
            ("base", _pcm_len(good_durations["hello world"])): "hello world",
            ("base", _pcm_len(good_durations["the quick brown fox"])): "the quick brown fox",
            ("base", _pcm_len(bad_durations["hello world"])): "hello word",
            ("base", _pcm_len(bad_durations["the quick brown fox"])): "the quick brown box",
        }
    )

    results = run_tts_comparison(TEXTS, [good, bad], stt_backend, tmp_path / "audio")

    assert [r.provider for r in results] == ["pyttsx3", "gtts"]
    good_result, bad_result = results
    assert good_result.voice == "default"
    assert good_result.mean_word_error_rate == pytest.approx(0.0)
    assert bad_result.mean_word_error_rate == pytest.approx((0.5 + 0.25) / 2)
    assert good_result.mean_word_error_rate < bad_result.mean_word_error_rate
    assert len(good_result.texts) == len(bad_result.texts) == 2
    assert good_result.texts[0].reference_text == "hello world"
    assert good_result.texts[0].hypothesis_text == "hello world"
    assert all(Path(t.audio_path).is_file() for t in good_result.texts + bad_result.texts)


def test_empty_text_set_is_rejected(good_variant):
    good, _ = good_variant
    with pytest.raises(TtsComparisonError):
        run_tts_comparison([], [good], _PerClipSttBackend({}), "unused")


def test_no_variants_is_rejected():
    with pytest.raises(TtsComparisonError):
        run_tts_comparison(TEXTS, [], _PerClipSttBackend({}), "unused")


def test_tts_backend_failure_propagates_instead_of_being_scored_silently(tmp_path):
    class _FailingTTSBackend:
        def synthesize(self, text, *, voice):
            raise RuntimeError("provider crashed")

    variant = TtsVariant(provider="pyttsx3", voice=None, backend=_FailingTTSBackend())

    with pytest.raises(SynthesisError):
        run_tts_comparison(TEXTS, [variant], _PerClipSttBackend({}), tmp_path / "audio")


def test_stt_backend_failure_propagates_instead_of_being_scored_silently(tmp_path, good_variant):
    good, _ = good_variant

    class _FailingSttBackend:
        def transcribe(self, samples, *, sample_rate, channels, model_size):
            raise RuntimeError("model crashed")

    with pytest.raises(TranscriptionError):
        run_tts_comparison(TEXTS, [good], _FailingSttBackend(), tmp_path / "audio")


def test_results_persist_per_variant_atomically(tmp_path, good_variant):
    good, durations = good_variant
    stt_backend = _PerClipSttBackend(
        {
            ("base", _pcm_len(durations["hello world"])): "hello world",
            ("base", _pcm_len(durations["the quick brown fox"])): "the quick brown fox",
        }
    )
    results = run_tts_comparison(TEXTS, [good], stt_backend, tmp_path / "audio")

    out = write_tts_comparison_results(results, tmp_path / "results")

    assert out.is_file()
    assert not list((tmp_path / "results").glob("*.tmp"))
    assert load_tts_comparison_results(out) == results
