"""TTS provider/voice comparison harness (README.md Phase 4, issue #13).

Comparing synthesized speech quality has no automatic ground truth on its
own, so this harness uses the standard human-free proxy: synthesize each
reference text with `voice_core.tts.TextToSpeech` under every requested
(provider, voice) combination, transcribe the resulting audio back with
`voice_core.stt.SpeechToText`, and score intelligibility as word error rate
against the original reference text. A provider/voice that round-trips
cleanly scores near zero WER; one that mangles the text scores high --
exactly mirroring how `comparison.py` compares Whisper model sizes, just
with the synthesis step added in front.

Each (provider, voice) combination carries its own `TTSBackend` instance
(mirrors `voice_core.tts.TextToSpeech`, whose backend is provider-specific --
a real deployment wires a different backend per provider, e.g. one built on
`pyttsx3`, one on `gTTS`, one on a cloud provider). No real TTS engine,
Whisper model, or GPU is available in this environment (or CI), so
`run_tts_comparison` takes those backends as fakes in tests, production
code supplies real ones.
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
from voice_core.tts import TextToSpeech, TTSBackend

__all__ = [
    "TtsComparisonError",
    "TtsVariant",
    "TextResult",
    "VoiceVariantResult",
    "run_tts_comparison",
    "write_tts_comparison_results",
    "load_tts_comparison_results",
]


class TtsComparisonError(ValueError):
    pass


@dataclass(frozen=True)
class TtsVariant:
    """One provider/voice combination to evaluate, with the backend that
    actually performs synthesis for it."""

    provider: str
    voice: str | None
    backend: TTSBackend


@dataclass(frozen=True)
class TextResult:
    """One reference text's round-trip outcome under one variant."""

    reference_text: str
    hypothesis_text: str
    word_error_rate: float
    audio_path: str
    duration_seconds: float


@dataclass(frozen=True)
class VoiceVariantResult:
    """One provider/voice combination's full outcome across the text set."""

    provider: str
    voice: str | None
    stt_model_size: str
    texts: tuple[TextResult, ...]
    mean_word_error_rate: float
    total_duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "voice": self.voice,
            "stt_model_size": self.stt_model_size,
            "mean_word_error_rate": self.mean_word_error_rate,
            "total_duration_seconds": self.total_duration_seconds,
            "texts": [asdict(t) for t in self.texts],
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "VoiceVariantResult":
        return cls(
            provider=raw["provider"],
            voice=raw["voice"],
            stt_model_size=raw["stt_model_size"],
            texts=tuple(TextResult(**t) for t in raw["texts"]),
            mean_word_error_rate=raw["mean_word_error_rate"],
            total_duration_seconds=raw["total_duration_seconds"],
        )


def run_tts_comparison(
    texts: Sequence[str],
    variants: Sequence[TtsVariant],
    stt_backend: WhisperBackend,
    audio_dir: str | os.PathLike[str],
    *,
    model_size: str = "base",
) -> tuple[VoiceVariantResult, ...]:
    """Synthesize every text under every variant and score the round trip.

    A synthesis or transcription failure on one clip propagates rather than
    being recorded as a fabricated result -- a harness that quietly scores a
    broken backend call as "100% wrong" would look identical to a genuinely
    bad provider/voice, hiding the real problem (portfolio rule: a wrong
    result produced silently is worse than a crash).
    """
    if not texts:
        raise TtsComparisonError("cannot compare an empty text set")
    if not variants:
        raise TtsComparisonError("at least one provider/voice variant is required")

    audio_root = Path(audio_dir)
    audio_root.mkdir(parents=True, exist_ok=True)

    stt = SpeechToText(stt_backend, model_size=model_size)

    results: list[VoiceVariantResult] = []
    for variant_index, variant in enumerate(variants):
        tts = TextToSpeech(variant.backend, provider=variant.provider, voice=variant.voice)
        text_results: list[TextResult] = []
        for text_index, text in enumerate(texts):
            out_path = audio_root / f"variant{variant_index}_text{text_index}.wav"
            synthesis = tts.synthesize(text, out_path)
            audio = FileAudioSource(synthesis.audio_path).capture()
            hypothesis = stt.transcribe(audio).text
            text_results.append(
                TextResult(
                    reference_text=text,
                    hypothesis_text=hypothesis,
                    word_error_rate=jiwer.wer(text, hypothesis),
                    audio_path=synthesis.audio_path,
                    duration_seconds=round(audio.duration_seconds, 3),
                )
            )
        results.append(
            VoiceVariantResult(
                provider=variant.provider,
                voice=variant.voice,
                stt_model_size=model_size,
                texts=tuple(text_results),
                mean_word_error_rate=statistics.mean(
                    t.word_error_rate for t in text_results
                ),
                total_duration_seconds=round(
                    sum(t.duration_seconds for t in text_results), 3
                ),
            )
        )
    return tuple(results)


def write_tts_comparison_results(
    results: Sequence[VoiceVariantResult], out_dir: str | os.PathLike[str]
) -> Path:
    """Write `results` to `<out_dir>/tts_comparison_results.json` atomically
    (portfolio rule 1)."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    target = out_root / "tts_comparison_results.json"
    payload = {
        "variants": [{"provider": r.provider, "voice": r.voice} for r in results],
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


def load_tts_comparison_results(
    path: str | os.PathLike[str],
) -> tuple[VoiceVariantResult, ...]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(VoiceVariantResult.from_dict(r) for r in raw["results"])
