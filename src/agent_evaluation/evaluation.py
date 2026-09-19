"""Phase 3: recording, transcription, LLM-as-judge scoring, and per-turn
latency for a completed persona call (issue #11).

Ties the Phase 2 simulated-caller loop's output (`caller.CallResult`) to
`voice_core`'s shared audio/STT/latency instrumentation and this feature
app's own `recording`/`judge` backends: every spoken line in the call is
recorded and transcribed (timed via the Phase 1 `LatencyTracker` every
other feature app uses), the transcript is scored by an `LLMJudge` against
the persona's own success criteria, and both are returned together as one
`EvaluationResult` a report can render.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from voice_core.latency import LatencyTracker
from voice_core.stt import SpeechToText, TranscriptionResult

from .caller import CallResult
from .judge import JudgeVerdict, LLMJudge
from .persona import Persona
from .recording import CallRecorder

__all__ = [
    "TurnTranscript",
    "EvaluationResult",
    "evaluate_call",
]


@dataclass(frozen=True)
class TurnTranscript:
    """One recorded+transcribed line of a call: which turn it belongs to,
    who spoke it, the original text, and the transcription that came back
    out of the record -> transcribe round trip."""

    turn_index: int
    speaker: str  # "caller" | "agent"
    text: str
    transcription: TranscriptionResult


@dataclass(frozen=True)
class EvaluationResult:
    """One call's full Phase 3 outcome: the underlying `CallResult`, the
    turn-by-turn transcript, and the judge's verdict.

    `verdict` is `None` when the call never produced a transcript worth
    scoring (nothing was said, e.g. it failed to connect) -- scoring an
    empty transcript would just be a confusing always-fail verdict rather
    than useful signal.
    """

    call: CallResult
    transcript: tuple[TurnTranscript, ...]
    verdict: JudgeVerdict | None
    run_id: str


def evaluate_call(
    persona: Persona,
    call: CallResult,
    *,
    recorder: CallRecorder,
    stt: SpeechToText,
    judge: LLMJudge,
    latency: LatencyTracker,
    run_id: str,
) -> EvaluationResult:
    """Record, transcribe, and score every spoken line of `call`.

    Only turns that actually produced text are recorded -- a turn with no
    `caller_utterance`/`agent_response` (e.g. the agent never responded)
    has nothing to record. Each record/transcribe step is timed as its own
    stage on `latency` under `run_id`, keyed by turn and speaker, so a
    report can break latency down per line rather than only per call.
    """
    transcripts: list[TurnTranscript] = []

    for turn in call.turns:
        for speaker, text in (
            ("caller", turn.caller_utterance),
            ("agent", turn.agent_response),
        ):
            if not text:
                continue

            with latency.stage(run_id, f"record:{speaker}:{turn.turn_index}"):
                audio = recorder.record_utterance(text)
            with latency.stage(run_id, f"transcribe:{speaker}:{turn.turn_index}"):
                result = stt.transcribe(audio)

            transcripts.append(
                TurnTranscript(
                    turn_index=turn.turn_index, speaker=speaker, text=text, transcription=result
                )
            )

    if not transcripts:
        return EvaluationResult(
            call=call, transcript=(), verdict=None, run_id=run_id
        )

    full_transcript = "\n".join(
        f"{t.speaker}: {t.transcription.text}" for t in transcripts
    )
    with latency.stage(run_id, "judge"):
        verdict = judge.score(full_transcript, persona)

    return EvaluationResult(
        call=call, transcript=tuple(transcripts), verdict=verdict, run_id=run_id
    )
