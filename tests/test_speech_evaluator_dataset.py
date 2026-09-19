import json
import struct
import wave

import pytest

from speech_evaluator.dataset import (
    DatasetError,
    TranscriptError,
    assemble_dataset,
    load_manifest,
    write_manifest,
)


def _write_silent_wav(path, seconds=0.5, rate=8000):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<" + "h" * n, *([0] * n)))


@pytest.fixture
def sample_dataset_dir(tmp_path):
    audio = tmp_path / "audio"
    audio.mkdir()
    _write_silent_wav(audio / "clip1.wav", seconds=0.5)
    _write_silent_wav(audio / "clip2.wav", seconds=1.0)
    (audio / "clip1.txt").write_text("hello world", encoding="utf-8")
    (audio / "clip2.txt").write_text("the quick brown fox", encoding="utf-8")
    return audio


def test_assemble_happy_path(sample_dataset_dir):
    entries = assemble_dataset(sample_dataset_dir, language="en")
    assert [e.transcript for e in entries] == ["hello world", "the quick brown fox"]
    assert all(e.language == "en" for e in entries)
    assert entries[0].duration_seconds == pytest.approx(0.5, abs=0.01)
    assert entries[1].duration_seconds == pytest.approx(1.0, abs=0.01)
    assert entries[0].audio_path.endswith("clip1.wav")


def test_manifest_roundtrip(sample_dataset_dir, tmp_path):
    entries = assemble_dataset(sample_dataset_dir)
    out = write_manifest(entries, tmp_path / "ds")
    assert out.is_file()
    data = json.loads(out.read_text())
    assert data["count"] == 2
    assert data["total_duration_seconds"] == pytest.approx(1.5, abs=0.02)
    assert [e.transcript for e in load_manifest(out)] == [
        "hello world",
        "the quick brown fox",
    ]
    assert not list((tmp_path / "ds").glob("*.tmp"))


def test_missing_transcript_is_hard_error(sample_dataset_dir):
    (sample_dataset_dir / "clip2.txt").unlink()
    with pytest.raises(TranscriptError):
        assemble_dataset(sample_dataset_dir)


def test_empty_transcript_rejected(sample_dataset_dir):
    (sample_dataset_dir / "clip1.txt").write_text("   \n", encoding="utf-8")
    with pytest.raises(TranscriptError):
        assemble_dataset(sample_dataset_dir)


def test_non_wav_audio_is_ignored(tmp_path):
    (tmp_path / "a.mp3").write_bytes(b"not really mp3")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    with pytest.raises(DatasetError):  # no .wav -> DatasetError
        assemble_dataset(tmp_path)


def test_corrupt_wav_raises_dataset_error(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"RIFFxxxxWAVEjunk")
    (tmp_path / "bad.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(DatasetError):
        assemble_dataset(tmp_path)


def test_missing_audio_dir_is_hard_error(tmp_path):
    with pytest.raises(DatasetError):
        assemble_dataset(tmp_path / "nope")


def test_missing_explicit_transcript_dir_is_hard_error(sample_dataset_dir, tmp_path):
    with pytest.raises(DatasetError):
        assemble_dataset(sample_dataset_dir, tmp_path / "nope-transcripts")
