"""Reference dataset assembly for the Whisper size/version comparison
harness (README.md Phase 4, issue #12).

Ported from the archived `speech-model-evaluator` repo's `src/sme/dataset.py`,
narrowed to WAV only so every entry can actually be transcribed through this
suite's own `voice_core.audio.FileAudioSource` -> `voice_core.stt.SpeechToText`
(the original also accepted MP3/FLAC via `mutagen`, which this suite's audio
layer doesn't read). The manifest shape (`count`, `total_duration_seconds`,
entries with `audio_path`/`transcript`/`language`/`duration_seconds`) is
unchanged, so tooling built against the original format still works.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from voice_core.audio import AudioCaptureError, FileAudioSource

AUDIO_EXTS = {".wav"}


class DatasetError(Exception):
    pass


class TranscriptError(DatasetError):
    pass


@dataclass(frozen=True)
class DatasetEntry:
    audio_path: str
    transcript: str
    language: str
    duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _read_transcript(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise TranscriptError(f"cannot read transcript {path}: {exc}") from exc
    if not text:
        raise TranscriptError(f"transcript {path} is empty")
    return text


def assemble_dataset(
    audio_dir: str | os.PathLike[str],
    transcript_dir: str | os.PathLike[str] | None = None,
    *,
    language: str = "en",
) -> list[DatasetEntry]:
    """Pair each `.wav` file under `audio_dir` with `<stem>.txt`.

    `transcript_dir` defaults to `audio_dir`. A user-supplied directory that
    does not exist is a hard error (strict on explicit input, per portfolio
    rule 7). A missing/empty transcript is also a hard error rather than a
    silent skip -- that would quietly shrink the benchmark set and make two
    runs incomparable.
    """
    audio_root = Path(audio_dir)
    if not audio_root.is_dir():
        raise DatasetError(f"audio directory not found: {audio_root}")
    transcript_root = Path(transcript_dir) if transcript_dir is not None else audio_root
    if not transcript_root.is_dir():
        raise DatasetError(f"transcript directory not found: {transcript_root}")

    audio_files = sorted(p for p in audio_root.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    if not audio_files:
        raise DatasetError(f"no .wav files under {audio_root}")

    entries: list[DatasetEntry] = []
    for audio_path in audio_files:
        transcript_path = transcript_root / f"{audio_path.stem}.txt"
        if not transcript_path.is_file():
            raise TranscriptError(
                f"no transcript for {audio_path.name} (expected {transcript_path})"
            )
        try:
            buffer = FileAudioSource(audio_path).capture()
        except AudioCaptureError as exc:
            raise DatasetError(f"cannot read WAV {audio_path}: {exc}") from exc
        entries.append(
            DatasetEntry(
                audio_path=str(audio_path.resolve()),
                transcript=_read_transcript(transcript_path),
                language=language,
                duration_seconds=round(buffer.duration_seconds, 3),
            )
        )
    return entries


def write_manifest(entries: list[DatasetEntry], out_dir: str | os.PathLike[str]) -> Path:
    """Write `entries` to `<out_dir>/manifest.json` atomically (portfolio
    rule 1) so an interrupted run never leaves a half-written manifest."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "manifest.json"
    payload = {
        "count": len(entries),
        "total_duration_seconds": round(sum(e.duration_seconds for e in entries), 3),
        "entries": [e.to_dict() for e in entries],
    }
    data = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")

    fd, tmp = tempfile.mkstemp(dir=out_root, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
    return target


def load_manifest(path: str | os.PathLike[str]) -> list[DatasetEntry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DatasetEntry(**entry) for entry in raw["entries"]]
