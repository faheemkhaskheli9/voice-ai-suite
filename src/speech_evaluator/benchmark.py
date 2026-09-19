"""Latency benchmarking and generation cost tracking for the Whisper and TTS
comparison harnesses (README.md Phase 4, issue #14).

Wraps `comparison.run_comparison` (STT) and `tts_comparison.run_tts_comparison`
(TTS) with the Phase 1 shared `voice_core.latency.LatencyTracker` -- one
timed run per model size / provider-voice combination -- and pairs each
timing with an estimated generation cost, producing one `BenchmarkEntry`
shape for both categories so they land in a single comparable report
(this issue's acceptance criterion).

No real provider billing API is available in this environment (or CI), so
cost is a documented per-second-of-audio estimate rather than a live
lookup: `DEFAULT_STT_COST_PER_SECOND`/`DEFAULT_TTS_COST_PER_SECOND` are
illustrative rates loosely modeled on typical published hosted-STT/TTS
per-minute pricing, converted to per-second -- not a real billing
integration. A caller may override per label via `cost_rates`.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from voice_core.latency import LatencyTracker
from voice_core.stt import WhisperBackend
from voice_core.tts import TTSBackend

from .comparison import run_comparison
from .dataset import DatasetEntry
from .tts_comparison import TtsVariant, run_tts_comparison

__all__ = [
    "BenchmarkError",
    "BenchmarkEntry",
    "DEFAULT_STT_COST_PER_SECOND",
    "DEFAULT_TTS_COST_PER_SECOND",
    "benchmark_stt_comparison",
    "benchmark_tts_comparison",
    "write_benchmark_results",
    "load_benchmark_results",
]

#: Illustrative estimate only -- see module docstring. Loosely modeled on
#: hosted Whisper-API pricing (~$0.006/minute), converted to per second.
DEFAULT_STT_COST_PER_SECOND = 0.0001

#: Illustrative estimate only -- see module docstring. Loosely modeled on
#: typical hosted TTS pricing per second of generated audio.
DEFAULT_TTS_COST_PER_SECOND = 0.00025


class BenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class BenchmarkEntry:
    """One model-size or provider/voice combination's benchmark outcome --
    the same shape for both the STT and TTS harnesses so the two are
    directly comparable in one report."""

    category: str  # "stt" or "tts"
    label: str  # model_size, or "provider:voice"
    mean_word_error_rate: float
    audio_duration_seconds: float
    latency_seconds: float
    estimated_cost_usd: float


def _cost_for(label: str, category_default: float, cost_rates: Mapping[str, float] | None) -> float:
    if cost_rates is not None and label in cost_rates:
        return cost_rates[label]
    return category_default


def benchmark_stt_comparison(
    entries: Sequence[DatasetEntry],
    model_sizes: Sequence[str],
    backend: WhisperBackend,
    *,
    tracker: LatencyTracker | None = None,
    cost_rates: Mapping[str, float] | None = None,
) -> tuple[BenchmarkEntry, ...]:
    """Benchmark each Whisper model size in `model_sizes` individually so
    `tracker` records one run's wall-clock latency per size."""
    if not model_sizes:
        raise BenchmarkError("at least one model size is required")

    tracker = tracker if tracker is not None else LatencyTracker()
    results: list[BenchmarkEntry] = []
    for model_size in model_sizes:
        run_id = f"stt:{model_size}"
        with tracker.stage(run_id, "stt"):
            (variant,) = run_comparison(entries, [model_size], backend)
        latency = tracker.get_run(run_id).total_seconds
        rate = _cost_for(model_size, DEFAULT_STT_COST_PER_SECOND, cost_rates)
        results.append(
            BenchmarkEntry(
                category="stt",
                label=model_size,
                mean_word_error_rate=variant.mean_word_error_rate,
                audio_duration_seconds=variant.total_duration_seconds,
                latency_seconds=round(latency, 6),
                estimated_cost_usd=round(variant.total_duration_seconds * rate, 6),
            )
        )
    return tuple(results)


def benchmark_tts_comparison(
    texts: Sequence[str],
    variants: Sequence[TtsVariant],
    stt_backend: WhisperBackend,
    audio_dir: str | os.PathLike[str],
    *,
    model_size: str = "base",
    tracker: LatencyTracker | None = None,
    cost_rates: Mapping[str, float] | None = None,
) -> tuple[BenchmarkEntry, ...]:
    """Benchmark each TTS provider/voice `variant` individually so `tracker`
    records one run's wall-clock latency per variant."""
    if not variants:
        raise BenchmarkError("at least one provider/voice variant is required")

    tracker = tracker if tracker is not None else LatencyTracker()
    results: list[BenchmarkEntry] = []
    for variant in variants:
        label = f"{variant.provider}:{variant.voice or 'default'}"
        run_id = f"tts:{label}"
        with tracker.stage(run_id, "tts"):
            (result,) = run_tts_comparison(
                texts, [variant], stt_backend, audio_dir, model_size=model_size
            )
        latency = tracker.get_run(run_id).total_seconds
        rate = _cost_for(label, DEFAULT_TTS_COST_PER_SECOND, cost_rates)
        results.append(
            BenchmarkEntry(
                category="tts",
                label=label,
                mean_word_error_rate=result.mean_word_error_rate,
                audio_duration_seconds=result.total_duration_seconds,
                latency_seconds=round(latency, 6),
                estimated_cost_usd=round(result.total_duration_seconds * rate, 6),
            )
        )
    return tuple(results)


def write_benchmark_results(
    entries: Sequence[BenchmarkEntry], out_dir: str | os.PathLike[str]
) -> Path:
    """Write `entries` to `<out_dir>/benchmark_results.json` atomically
    (portfolio rule 1)."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "benchmark_results.json"
    payload = {"entries": [asdict(e) for e in entries]}
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


def load_benchmark_results(path: str | os.PathLike[str]) -> tuple[BenchmarkEntry, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(BenchmarkEntry(**e) for e in raw["entries"])
