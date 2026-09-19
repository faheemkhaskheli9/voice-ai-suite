"""LLM-as-judge scoring for a completed, transcribed persona call (issue #11).

Scores a call transcript against the persona's own `success_criteria`
rather than a hardcoded rubric, so the same judge works for every persona
script without code changes. No real LLM provider is available in this
environment (or CI), so `LLMJudge` talks to a pluggable `JudgeBackend`
instead of calling an LLM SDK directly -- production code supplies a real
backend (a prompted call to an LLM provider); tests and this module's
default supply a fake one, mirroring the `WhisperBackend`/`TTSBackend`
pattern in `voice_core`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .persona import Persona


class JudgeError(ValueError):
    """Raised when a transcript cannot be judged (backend failure, an
    unusable verdict) -- carries a clear `reason` instead of letting the
    caller see a raw backend exception or a crash."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class NoJudgeBackendAvailable(JudgeError):
    """Raised by `LLMJudge` when no backend was supplied and no
    network-backed default is available in this environment (this is
    always true in CI and in this scaffold, per the Definition of Done: no
    paid APIs)."""

    def __init__(self):
        super().__init__(
            "No judge backend available. Pass a JudgeBackend "
            "(a real one wrapping an LLM provider for a real app, a fake "
            "one for tests)."
        )


@dataclass(frozen=True)
class JudgeVerdict:
    """One transcript's scored outcome: enough detail for a report to show
    *why* a call passed/failed, not just the flag."""

    passed: bool
    score: float
    rationale: str
    criteria_met: tuple[str, ...] = field(default_factory=tuple)
    criteria_missed: tuple[str, ...] = field(default_factory=tuple)


class JudgeBackend(Protocol):
    """What a real LLM-judge implementation must provide. Kept tiny and
    framework-free so it can be backed by any LLM provider or a test fake
    interchangeably."""

    def judge(self, transcript: str, success_criteria: list[str], goal: str) -> dict:
        """Score `transcript` against `success_criteria`/`goal` and return a
        dict with at least a `score` (0.0-1.0) and `rationale`; may also
        include `criteria_met`/`criteria_missed` lists."""


class FakeJudgeBackend:
    """Deterministic keyword-overlap judge for tests and CI: a criterion is
    "met" if any of its significant (len > 3) words appears in the
    transcript (case-insensitive). No real LLM call, but real transcript-
    dependent scoring -- the same call scores differently against a
    different transcript, so tests can exercise both pass and fail paths."""

    #: Excluded from keyword matching because `evaluate_call` always
    #: prefixes every transcript line with one of these as a speaker label
    #: (e.g. "agent: ..."), which would otherwise make any criterion
    #: mentioning "agent"/"caller" trivially match every transcript.
    _SPEAKER_LABELS = frozenset({"agent", "caller"})

    def judge(self, transcript: str, success_criteria: list[str], goal: str) -> dict:
        transcript_lower = transcript.lower()
        met: list[str] = []
        missed: list[str] = []
        for criterion in success_criteria:
            words = [
                w
                for w in criterion.lower().split()
                if len(w) > 3 and w not in self._SPEAKER_LABELS
            ]
            if words and any(w in transcript_lower for w in words):
                met.append(criterion)
            else:
                missed.append(criterion)

        score = len(met) / len(success_criteria) if success_criteria else 0.0
        rationale = (
            f"{len(met)}/{len(success_criteria)} success criteria matched in the transcript."
        )
        return {
            "score": score,
            "rationale": rationale,
            "criteria_met": met,
            "criteria_missed": missed,
        }


class LLMJudge:
    """The one LLM-as-judge interface the evaluation pipeline calls.

    `pass_threshold` is a constructor argument (never hardcoded), so
    tightening/loosening the bar for "pass" is a config change, not a code
    change in any caller.
    """

    def __init__(self, backend: JudgeBackend | None = None, *, pass_threshold: float = 0.7):
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError(f"pass_threshold must be within [0, 1], got {pass_threshold!r}")
        self._backend = backend
        self.pass_threshold = pass_threshold

    def score(self, transcript: str, persona: Persona) -> JudgeVerdict:
        if self._backend is None:
            raise NoJudgeBackendAvailable()
        if not transcript.strip():
            raise JudgeError("Cannot judge an empty transcript.")

        try:
            raw = self._backend.judge(transcript, persona.success_criteria, persona.goal)
        except Exception as exc:  # noqa: BLE001 - any backend failure -> one clear error
            raise JudgeError(f"Judge backend failed: {exc}") from exc

        score = raw.get("score")
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            raise JudgeError("Judge backend returned no usable 'score' within [0, 1].")
        rationale = raw.get("rationale")
        if not isinstance(rationale, str) or not rationale:
            raise JudgeError("Judge backend returned no usable 'rationale'.")

        return JudgeVerdict(
            passed=score >= self.pass_threshold,
            score=float(score),
            rationale=rationale,
            criteria_met=tuple(raw.get("criteria_met", ())),
            criteria_missed=tuple(raw.get("criteria_missed", ())),
        )
