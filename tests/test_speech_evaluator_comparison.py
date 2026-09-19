import struct
import wave

import pytest

from speech_evaluator.comparison import (
    ComparisonError,
    load_comparison_results,
    run_comparison,
    write_comparison_results,
)
from speech_evaluator.dataset import DatasetEntry
from voice_core.stt import TranscriptionError

SAMPLE_RATE = 8000


def _write_silent_wav(path, seconds, rate=SAMPLE_RATE):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


def _pcm_len(seconds, rate=SAMPLE_RATE):
    """Bytes-of-PCM a clip of this length decodes to -- lets a fake backend
    tell clips apart the same way `voice_core.audio.AudioBuffer` would (mono,
    16-bit), without needing real speech content."""
    return int(seconds * rate) * 2


@pytest.fixture
def entries(tmp_path):
    clip1 = tmp_path / "clip1.wav"
    clip2 = tmp_path / "clip2.wav"
    _write_silent_wav(clip1, seconds=0.5)
    _write_silent_wav(clip2, seconds=1.0)
    return [
        DatasetEntry(
            audio_path=str(clip1), transcript="hello world", language="en", duration_seconds=0.5
        ),
        DatasetEntry(
            audio_path=str(clip2),
            transcript="the quick brown fox",
            language="en",
            duration_seconds=1.0,
        ),
    ]


class _PerClipBackend:
    """Returns a canned transcription keyed by (model_size, clip length),
    so a size sweep can return genuinely different, clip-specific text --
    standing in for a real model that gets some clips right and others
    wrong depending on size."""

    def __init__(self, responses: dict[tuple[str, int], str]):
        self._responses = responses

    def transcribe(self, samples: bytes, *, sample_rate: int, channels: int, model_size: str) -> dict:
        return {"text": self._responses[(model_size, len(samples))]}


def test_compares_multiple_model_sizes(entries):
    clip1_len, clip2_len = _pcm_len(0.5), _pcm_len(1.0)
    backend = _PerClipBackend(
        {
            ("tiny", clip1_len): "hello word",  # 1 substitution / 2 words
            ("tiny", clip2_len): "the quick brown box",  # 1 substitution / 4 words
            ("large-v3", clip1_len): "hello world",  # exact
            ("large-v3", clip2_len): "the quick brown fox",  # exact
        }
    )

    results = run_comparison(entries, ["tiny", "large-v3"], backend)

    assert [r.model_size for r in results] == ["tiny", "large-v3"]
    tiny, large = results
    assert tiny.mean_word_error_rate == pytest.approx((0.5 + 0.25) / 2)
    assert large.mean_word_error_rate == pytest.approx(0.0)
    assert tiny.mean_word_error_rate > large.mean_word_error_rate
    assert len(tiny.entries) == len(large.entries) == 2
    assert tiny.entries[0].reference == "hello world"
    assert tiny.entries[0].hypothesis == "hello word"
    assert tiny.total_duration_seconds == pytest.approx(1.5, abs=0.02)


def test_at_least_two_sizes_run_against_the_same_audio_set(entries):
    clip1_len, clip2_len = _pcm_len(0.5), _pcm_len(1.0)
    backend = _PerClipBackend(
        {
            (size, length): "hello world" if length == clip1_len else "the quick brown fox"
            for size in ("tiny", "base", "small")
            for length in (clip1_len, clip2_len)
        }
    )

    results = run_comparison(entries, ["tiny", "base", "small"], backend)

    assert len(results) == 3
    audio_paths_by_size = [tuple(e.audio_path for e in r.entries) for r in results]
    assert audio_paths_by_size[0] == audio_paths_by_size[1] == audio_paths_by_size[2]
    assert all(r.mean_word_error_rate == pytest.approx(0.0) for r in results)


def test_empty_dataset_is_rejected():
    with pytest.raises(ComparisonError):
        run_comparison([], ["tiny"], _PerClipBackend({}))


def test_no_model_sizes_is_rejected(entries):
    with pytest.raises(ComparisonError):
        run_comparison(entries, [], _PerClipBackend({}))


def test_backend_failure_propagates_instead_of_being_scored_silently(entries):
    class _FailingBackend:
        def transcribe(self, samples, *, sample_rate, channels, model_size):
            raise RuntimeError("model crashed")

    with pytest.raises(TranscriptionError):
        run_comparison(entries, ["tiny"], _FailingBackend())


def test_results_persist_per_model_variant_atomically(entries, tmp_path):
    clip1_len, clip2_len = _pcm_len(0.5), _pcm_len(1.0)
    backend = _PerClipBackend(
        {
            ("tiny", clip1_len): "hello world",
            ("tiny", clip2_len): "the quick brown fox",
        }
    )
    results = run_comparison(entries, ["tiny"], backend)

    out = write_comparison_results(results, tmp_path / "results")

    assert out.is_file()
    assert not list((tmp_path / "results").glob("*.tmp"))
    assert load_comparison_results(out) == results
