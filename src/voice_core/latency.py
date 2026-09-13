"""
Shared latency instrumentation for every voice feature app (README.md
Section 2/4, issue #4): one `LatencyTracker` that every feature wraps its
STT/LLM/TTS calls with instead of hand-rolling its own timer, so stage
timings are recorded and retrievable the same way everywhere -- including
by the Phase 4 Speech Model Evaluator feature, which is meant to consume
exactly this per-run timing data when it compares providers.

Uses a monotonic clock (`time.perf_counter` by default) rather than
wall-clock time, so timings are immune to system clock adjustments, and a
context manager per stage so instrumentation overhead is just two clock
reads -- negligible next to a real STT/LLM/TTS call (this issue's
acceptance criterion that overhead not materially change measured
latency).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator


class UnknownRun(KeyError):
    """Raised when `get_run` is asked for a run id that was never
    recorded -- a clear error instead of a raw KeyError with no context."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        super().__init__(f"No latency recorded for run {run_id!r}.")


@dataclass(frozen=True)
class StageTiming:
    """One stage's (e.g. "stt", "llm", "tts") measured duration."""

    stage: str
    duration_seconds: float


@dataclass
class RunLatency:
    """One interaction's (e.g. one real-time-agent turn, one RAG-chatbot
    query) per-stage timings, in the order they were recorded."""

    run_id: str
    stages: list[StageTiming] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return sum(s.duration_seconds for s in self.stages)

    def get(self, stage: str) -> float | None:
        """Duration of `stage` if recorded, else `None` -- a feature app
        checking an optional stage (e.g. "llm" on a clip that skipped
        generation) shouldn't need a try/except for a stage that may not
        apply to every run."""
        for s in self.stages:
            if s.stage == stage:
                return s.duration_seconds
        return None


class LatencyTracker:
    """Records per-stage timings for one or more runs and makes them
    retrievable by `run_id` afterward -- the Speech Model Evaluator
    feature's acceptance criterion ("timing data is retrievable per run").
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._runs: dict[str, RunLatency] = {}

    @contextmanager
    def stage(self, run_id: str, stage: str) -> Iterator[None]:
        """Time one stage (e.g. "stt", "llm", "tts") of `run_id`. Safe to
        call for several stages of the same run_id, in any order; each
        call appends one `StageTiming` rather than overwriting a prior one
        for the same stage name, so a retried stage still shows up."""
        start = self._clock()
        try:
            yield
        finally:
            duration = self._clock() - start
            run = self._runs.setdefault(run_id, RunLatency(run_id=run_id))
            run.stages.append(StageTiming(stage=stage, duration_seconds=duration))

    def get_run(self, run_id: str) -> RunLatency:
        try:
            return self._runs[run_id]
        except KeyError:
            raise UnknownRun(run_id) from None

    def runs(self) -> tuple[RunLatency, ...]:
        """All recorded runs, oldest first -- what a feature app (e.g. the
        Speech Model Evaluator) iterates to build its comparison report."""
        return tuple(self._runs.values())
