import time

import pytest

from voice_core.latency import LatencyTracker, UnknownRun


def _fake_clock(*readings):
    values = iter(readings)
    return lambda: next(values)


def test_single_stage_records_duration():
    tracker = LatencyTracker(clock=_fake_clock(100.0, 100.25))

    with tracker.stage("run-1", "stt"):
        pass

    run = tracker.get_run("run-1")
    assert run.get("stt") == pytest.approx(0.25)


def test_multiple_stages_same_run_preserve_order_and_sum():
    # stt: 100.0 -> 100.1 (0.1s), llm: 100.1 -> 100.4 (0.3s), tts: 100.4 -> 100.5 (0.1s)
    tracker = LatencyTracker(clock=_fake_clock(100.0, 100.1, 100.1, 100.4, 100.4, 100.5))

    with tracker.stage("run-1", "stt"):
        pass
    with tracker.stage("run-1", "llm"):
        pass
    with tracker.stage("run-1", "tts"):
        pass

    run = tracker.get_run("run-1")
    assert [s.stage for s in run.stages] == ["stt", "llm", "tts"]
    assert run.get("stt") == pytest.approx(0.1)
    assert run.get("llm") == pytest.approx(0.3)
    assert run.get("tts") == pytest.approx(0.1)
    assert run.total_seconds == pytest.approx(0.5)


def test_get_returns_none_for_unrecorded_stage():
    tracker = LatencyTracker(clock=_fake_clock(0.0, 0.1))
    with tracker.stage("run-1", "stt"):
        pass

    run = tracker.get_run("run-1")
    assert run.get("llm") is None


def test_unknown_run_raises_clear_error():
    tracker = LatencyTracker()
    with pytest.raises(UnknownRun):
        tracker.get_run("never-recorded")


def test_runs_returns_all_recorded_runs_oldest_first():
    tracker = LatencyTracker(clock=_fake_clock(0.0, 0.1, 1.0, 1.2))

    with tracker.stage("run-1", "stt"):
        pass
    with tracker.stage("run-2", "stt"):
        pass

    ids = [r.run_id for r in tracker.runs()]
    assert ids == ["run-1", "run-2"]


def test_exception_inside_stage_still_records_timing_and_propagates():
    tracker = LatencyTracker(clock=_fake_clock(0.0, 0.2))

    with pytest.raises(RuntimeError, match="boom"):
        with tracker.stage("run-1", "stt"):
            raise RuntimeError("boom")

    run = tracker.get_run("run-1")
    assert run.get("stt") == pytest.approx(0.2)


def test_instrumentation_overhead_is_negligible_next_to_real_work():
    """Sanity check with the real monotonic clock: instrumenting a stage
    that does ~20ms of work should not materially inflate the measured
    duration -- the acceptance criterion that overhead not materially
    change measured latency."""
    tracker = LatencyTracker()
    work_seconds = 0.02

    with tracker.stage("run-1", "stt"):
        time.sleep(work_seconds)

    measured = tracker.get_run("run-1").get("stt")
    assert measured is not None
    # Generous tolerance for scheduler jitter in CI -- the point is the
    # tracker itself adds no material overhead, not sub-millisecond
    # precision.
    assert work_seconds <= measured <= work_seconds + 0.05
