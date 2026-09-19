import pytest

from agent_evaluation.judge import (
    FakeJudgeBackend,
    JudgeError,
    LLMJudge,
    NoJudgeBackendAvailable,
)
from agent_evaluation.persona import Persona


def _persona(**overrides) -> Persona:
    defaults = dict(
        name="tester",
        goal="Get a refund processed.",
        tone="calm",
        sample_utterances=["Hello.", "I need a refund."],
        success_criteria=["Agent processes the refund.", "Agent confirms the amount."],
    )
    defaults.update(overrides)
    return Persona(**defaults)


def test_transcript_meeting_all_criteria_passes():
    judge = LLMJudge(FakeJudgeBackend())
    persona = _persona()
    transcript = "caller: I need a refund.\nagent: I will process the refund for the amount."

    verdict = judge.score(transcript, persona)

    assert verdict.passed is True
    assert verdict.score == 1.0
    assert verdict.criteria_met == tuple(persona.success_criteria)
    assert verdict.criteria_missed == ()


def test_transcript_missing_all_criteria_fails():
    judge = LLMJudge(FakeJudgeBackend())
    persona = _persona()
    transcript = "caller: Hello.\nagent: Sorry, I cannot help with that."

    verdict = judge.score(transcript, persona)

    assert verdict.passed is False
    assert verdict.score == 0.0
    assert verdict.criteria_missed == tuple(persona.success_criteria)


def test_pass_threshold_is_configurable_not_hardcoded():
    transcript = "caller: I need a refund.\nagent: sorry, no."
    persona = _persona()

    lenient = LLMJudge(FakeJudgeBackend(), pass_threshold=0.4)
    strict = LLMJudge(FakeJudgeBackend(), pass_threshold=0.9)

    assert lenient.score(transcript, persona).passed is True
    assert strict.score(transcript, persona).passed is False


def test_invalid_pass_threshold_rejected_at_construction():
    with pytest.raises(ValueError, match="pass_threshold"):
        LLMJudge(FakeJudgeBackend(), pass_threshold=1.5)


def test_no_backend_raises_clear_error_instead_of_crashing():
    judge = LLMJudge()
    with pytest.raises(NoJudgeBackendAvailable):
        judge.score("some transcript", _persona())


def test_empty_transcript_raises_clear_error():
    judge = LLMJudge(FakeJudgeBackend())
    with pytest.raises(JudgeError, match="empty"):
        judge.score("   ", _persona())


def test_backend_failure_raises_judge_error_not_the_raw_exception():
    class BoomBackend:
        def judge(self, transcript, success_criteria, goal):
            raise RuntimeError("provider timed out")

    judge = LLMJudge(BoomBackend())
    with pytest.raises(JudgeError, match="Judge backend failed"):
        judge.score("some transcript", _persona())


def test_backend_returning_no_usable_score_raises_clear_error():
    class BadBackend:
        def judge(self, transcript, success_criteria, goal):
            return {"rationale": "no score field"}

    judge = LLMJudge(BadBackend())
    with pytest.raises(JudgeError, match="no usable 'score'"):
        judge.score("some transcript", _persona())


def test_backend_returning_out_of_range_score_raises_clear_error():
    class BadBackend:
        def judge(self, transcript, success_criteria, goal):
            return {"score": 1.5, "rationale": "oops"}

    judge = LLMJudge(BadBackend())
    with pytest.raises(JudgeError, match="no usable 'score'"):
        judge.score("some transcript", _persona())
