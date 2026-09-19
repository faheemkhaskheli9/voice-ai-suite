import io
import itertools
import struct
import wave

import pytest

from speech_evaluator.benchmark import (
    DEFAULT_STT_COST_PER_SECOND,
    DEFAULT_TTS_COST_PER_SECOND,
    BenchmarkError,
    benchmark_stt_comparison,
    benchmark_tts_comparison,
    load_benchmark_results,
    write_benchmark_results,
)
from speech_evaluator.dataset import DatasetEntry
from speech_evaluator.tts_comparison import TtsVariant
from voice_core.latency import LatencyTracker

SAMPLE_RATE = 8000


def _fake_clock(step: float = 0.5):
    """Deterministic monotonic clock: each call advances by `step`
    seconds, so a `LatencyTracker.stage` block always measures exactly
    `step` seconds regardless of how fast the fake backends actually run."""
    counter = itertools.count()

    def clock() -> float:
        return next(counter) * step

    return clock


def _write_silent_wav(path, seconds, rate=SAMPLE_RATE):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


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


class _PerClipSttBackend:
    def __init__(self, responses):
        self._responses = responses

    def transcribe(self, samples, *, sample_rate, channels, model_size):
        return {"text": self._responses[(model_size, len(samples))]}


class _FakeTTSBackend:
    def __init__(self, duration_by_text):
        self._durations = duration_by_text

    def synthesize(self, text, *, voice):
        return _wav_bytes(self._durations[text])


def test_benchmark_stt_comparison_records_latency_and_cost(tmp_path):
    clip = tmp_path / "clip.wav"
    _write_silent_wav(clip, seconds=1.0)
    entries = [DatasetEntry(audio_path=str(clip), transcript="hello world", language="en", duration_seconds=1.0)]
    backend = _PerClipSttBackend({("tiny", _pcm_len(1.0)): "hello world"})
    tracker = LatencyTracker(clock=_fake_clock())

    results = benchmark_stt_comparison(entries, ["tiny"], backend, tracker=tracker)

    assert len(results) == 1
    entry = results[0]
    assert entry.category == "stt"
    assert entry.label == "tiny"
    assert entry.mean_word_error_rate == pytest.approx(0.0)
    assert entry.audio_duration_seconds == pytest.approx(1.0)
    assert entry.latency_seconds == pytest.approx(0.5)
    assert entry.estimated_cost_usd == pytest.approx(1.0 * DEFAULT_STT_COST_PER_SECOND)


def test_benchmark_stt_comparison_rejects_empty_model_sizes(tmp_path):
    clip = tmp_path / "clip.wav"
    _write_silent_wav(clip, seconds=1.0)
    entries = [DatasetEntry(audio_path=str(clip), transcript="hi", language="en", duration_seconds=1.0)]

    with pytest.raises(BenchmarkError):
        benchmark_stt_comparison(entries, [], _PerClipSttBackend({}))


def test_benchmark_tts_comparison_records_latency_and_cost(tmp_path):
    durations = {"hello world": 0.5}
    variant = TtsVariant(provider="pyttsx3", voice="default", backend=_FakeTTSBackend(durations))
    backend = _PerClipSttBackend({("base", _pcm_len(0.5)): "hello world"})
    tracker = LatencyTracker(clock=_fake_clock())

    results = benchmark_tts_comparison(["hello world"], [variant], backend, tmp_path / "audio", tracker=tracker)

    assert len(results) == 1
    entry = results[0]
    assert entry.category == "tts"
    assert entry.label == "pyttsx3:default"
    assert entry.mean_word_error_rate == pytest.approx(0.0)
    assert entry.latency_seconds == pytest.approx(0.5)
    assert entry.estimated_cost_usd == pytest.approx(0.5 * DEFAULT_TTS_COST_PER_SECOND)


def test_benchmark_tts_comparison_rejects_no_variants():
    with pytest.raises(BenchmarkError):
        benchmark_tts_comparison(["hi"], [], _PerClipSttBackend({}), "unused")


def test_custom_cost_rate_overrides_default(tmp_path):
    clip = tmp_path / "clip.wav"
    _write_silent_wav(clip, seconds=1.0)
    entries = [DatasetEntry(audio_path=str(clip), transcript="hi", language="en", duration_seconds=1.0)]
    backend = _PerClipSttBackend({("tiny", _pcm_len(1.0)): "hi"})

    results = benchmark_stt_comparison(
        entries, ["tiny"], backend, cost_rates={"tiny": 0.5}
    )

    assert results[0].estimated_cost_usd == pytest.approx(0.5)


def test_stt_and_tts_benchmark_entries_are_comparable_and_persist(tmp_path):
    clip = tmp_path / "clip.wav"
    _write_silent_wav(clip, seconds=1.0)
    entries = [DatasetEntry(audio_path=str(clip), transcript="hi there", language="en", duration_seconds=1.0)]
    stt_backend = _PerClipSttBackend(
        {
            ("tiny", _pcm_len(1.0)): "hi there",
            ("base", _pcm_len(0.5)): "hi there",
        }
    )
    variant = TtsVariant(provider="gtts", voice=None, backend=_FakeTTSBackend({"hi there": 0.5}))

    stt_results = benchmark_stt_comparison(entries, ["tiny"], stt_backend)
    tts_results = benchmark_tts_comparison(["hi there"], [variant], stt_backend, tmp_path / "audio")
    combined = stt_results + tts_results

    assert {e.category for e in combined} == {"stt", "tts"}
    assert all(hasattr(e, "mean_word_error_rate") and hasattr(e, "latency_seconds") for e in combined)

    out = write_benchmark_results(combined, tmp_path / "results")

    assert out.is_file()
    assert not list((tmp_path / "results").glob("*.tmp"))
    assert load_benchmark_results(out) == combined
