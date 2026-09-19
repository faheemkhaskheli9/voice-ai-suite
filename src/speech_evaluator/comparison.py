"""Whisper size/version comparison harness (README.md Phase 4, issue #12).

Runs the same manifest (a set of `dataset.DatasetEntry`) through
`voice_core.stt.SpeechToText` once per requested Whisper model size, scores
each transcription against its reference transcript with word error rate
(`jiwer`), and persists one `ModelVariantResult` per size so a caller can
compare accuracy across sizes -- exactly the sweep
`voice_core/stt.py`'s module docstring calls out `SpeechToText`'s
constructor-argument `model_size` as existing for.

No real Whisper model/GPU is available in this environment (or CI), so
`run_comparison` takes a `WhisperBackend` the same way `SpeechToText` does --
production code supplies a real one, tests supply a fake one.
"""
from __future__ import annotations

import contextlib
import json
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import jiwer

from voice_core.audio import FileAudioSource
from voice_core.stt import SpeechToText, WhisperBackend

from .dataset import DatasetEntry

__all__ = [
    "ComparisonError",
    "EntryResult",
    "ModelVariantResult",
    "run_comparison",
    "write_comparison_results",
    "load_comparison_results",
]


class ComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class EntryResult:
    """One clip's outcome under one model size."""

    audio_path: str
    reference: str
    hypothesis: str
    word_error_rate: float
    duration_seconds: float


@dataclass(frozen=True)
class ModelVariantResult:
    """One Whisper model size's full outcome across the dataset."""

    model_size: str
    entries: tuple[EntryResult, ...]
    mean_word_error_rate: float
    total_duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "model_size": self.model_size,
            "mean_word_error_rate": self.mean_word_error_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "entries": [asdict(e) for e in self.entries],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelVariantResult":
        return cls(
            model_size=raw["model_size"],
            entries=tuple(EntryResult(**e) for e in raw["entries"]),
            mean_word_error_rate=raw["mean_word_error_rate"],
            total_duration_seconds=raw["total_duration_seconds"],
        )


def run_comparison(
    entries: Sequence[DatasetEntry],
    model_sizes: Sequence[str],
    backend: WhisperBackend,
) -> tuple[ModelVariantResult, ...]:
    """Transcribe every entry once per model size and score it against its
    reference transcript.

    A backend failure on one clip propagates rather than being recorded as a
    fabricated result -- a harness that quietly scores a broken backend call
    as "100% wrong" would look identical to a genuinely bad model, hiding the
    real problem (portfolio rule: a wrong result produced silently is worse
    than a crash).
    """
    if not entries:
        raise ComparisonError("cannot compare an empty dataset")
    if not model_sizes:
        raise ComparisonError("at least one model size is required")

    results: list[ModelVariantResult] = []
    for model_size in model_sizes:
        stt = SpeechToText(backend, model_size=model_size)
        entry_results: list[EntryResult] = []
        for entry in entries:
            audio = FileAudioSource(entry.audio_path).capture()
            hypothesis = stt.transcribe(audio).text
            entry_results.append(
                EntryResult(
                    audio_path=entry.audio_path,
                    reference=entry.transcript,
                    hypothesis=hypothesis,
                    word_error_rate=jiwer.wer(entry.transcript, hypothesis),
                    duration_seconds=entry.duration_seconds,
                )
            )
        results.append(
            ModelVariantResult(
                model_size=model_size,
                entries=tuple(entry_results),
                mean_word_error_rate=statistics.mean(
                    e.word_error_rate for e in entry_results
                ),
                total_duration_seconds=round(
                    sum(e.duration_seconds for e in entry_results), 3
                ),
            )
        )
    return tuple(results)


def write_comparison_results(
    results: Sequence[ModelVariantResult], out_dir: str | os.PathLike[str]
) -> Path:
    """Write `results` to `<out_dir>/comparison_results.json` atomically
    (portfolio rule 1)."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "comparison_results.json"
    payload = {
        "model_sizes": [r.model_size for r in results],
        "results": [r.to_dict() for r in results],
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


def load_comparison_results(
    path: str | os.PathLike[str],
) -> tuple[ModelVariantResult, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(ModelVariantResult.from_dict(r) for r in raw["results"])
