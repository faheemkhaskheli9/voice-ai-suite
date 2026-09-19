from agent_evaluation.caller import FakeVoiceRoomBackend, run_persona_call
from agent_evaluation.evaluation import evaluate_call
from agent_evaluation.judge import FakeJudgeBackend, LLMJudge
from agent_evaluation.persona import Persona
from agent_evaluation.recording import CallRecorder
from voice_core.latency import LatencyTracker
from voice_core.stt import SpeechToText


class _EchoRecordingBackend:
    """Encodes the spoken text directly into the 'audio' bytes, so a
    matching echo Whisper backend can decode it back out -- lets tests
    assert the recorded+transcribed text round-trips correctly without a
    real audio codec."""

    def record(self, text: str, *, sample_rate: int, channels: int) -> bytes:
        data = text.encode("utf-8")
        return data if len(data) % 2 == 0 else data + b"\x00"


class _EchoWhisperBackend:
    def transcribe(self, samples: bytes, *, sample_rate: int, channels: int, model_size: str) -> dict:
        return {"text": samples.rstrip(b"\x00").decode("utf-8")}


def _persona(**overrides) -> Persona:
    defaults = dict(
        name="tester",
        goal="Get a refund processed.",
        tone="calm",
        sample_utterances=["Hello.", "I need a refund."],
        success_criteria=["Agent confirms the issue has been resolved."],
    )
    defaults.update(overrides)
    return Persona(**defaults)


def _pipeline():
    return dict(
        recorder=CallRecorder(_EchoRecordingBackend()),
        stt=SpeechToText(_EchoWhisperBackend()),
        judge=LLMJudge(FakeJudgeBackend()),
        latency=LatencyTracker(),
    )


def test_completed_call_is_recorded_transcribed_and_scored():
    persona = _persona()
    backend = FakeVoiceRoomBackend(lambda utterance: "I confirm the issue has been resolved.")
    call = run_persona_call(persona, backend, "room-1")

    result = evaluate_call(persona, call, run_id="run-1", **_pipeline())

    assert result.verdict is not None
    assert result.verdict.passed is True
    # 2 turns * (caller line + agent response) = 4 recorded/transcribed lines
    assert len(result.transcript) == 4
    assert result.transcript[0].speaker == "caller"
    assert result.transcript[0].text == "Hello."
    assert result.transcript[0].transcription.text == "Hello."
    assert result.transcript[1].speaker == "agent"
    assert result.transcript[1].transcription.text == "I confirm the issue has been resolved."


def test_call_failing_success_criteria_is_flagged_fail():
    persona = _persona()
    backend = FakeVoiceRoomBackend(lambda utterance: "I cannot help with that.")
    call = run_persona_call(persona, backend, "room-1")

    result = evaluate_call(persona, call, run_id="run-2", **_pipeline())

    assert result.verdict is not None
    assert result.verdict.passed is False


def test_per_turn_latency_is_recorded_via_shared_instrumentation():
    persona = _persona()
    backend = FakeVoiceRoomBackend(lambda utterance: "ok")
    call = run_persona_call(persona, backend, "room-1")
    pipeline = _pipeline()

    evaluate_call(persona, call, run_id="run-3", **pipeline)

    run = pipeline["latency"].get_run("run-3")
    stage_names = [s.stage for s in run.stages]
    assert "record:caller:0" in stage_names
    assert "transcribe:caller:0" in stage_names
    assert "record:agent:0" in stage_names
    assert "transcribe:agent:0" in stage_names
    assert "judge" in stage_names
    assert run.total_seconds >= 0.0


def test_call_with_no_spoken_lines_skips_scoring_instead_of_a_hollow_verdict():
    persona = _persona()

    class _RefusingBackend(FakeVoiceRoomBackend):
        def connect(self, room_name: str) -> None:
            from agent_evaluation.caller import RoomClosedError

            raise RoomClosedError("server refused connection")

    refusing = _RefusingBackend(lambda utterance: "ok")
    call = run_persona_call(persona, refusing, "room-1")
    assert call.turns == ()  # sanity: nothing was ever said

    result = evaluate_call(persona, call, run_id="run-4", **_pipeline())

    assert result.transcript == ()
    assert result.verdict is None
