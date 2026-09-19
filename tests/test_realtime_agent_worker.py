import asyncio
import struct

import pytest

from realtime_agent.config import LiveKitConfig
from realtime_agent.dialogue import (
    FinalResponse,
    RunState,
    ToolCall,
    ToolCallStep,
    ToolLoopExceeded,
    ToolRegistry,
)
from realtime_agent.memory import ConversationMemory, Turn
from realtime_agent.room import (
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_PARTICIPANT_JOINED,
    FakeRoomClient,
    build_room_client,
)
from realtime_agent.worker import EVENT_INTERRUPTED, EVENT_TOOL_CALL, AgentWorker
from voice_core.audio import AudioBuffer, MicAudioSource
from voice_core.stt import SpeechToText
from voice_core.tts import TextToSpeech


class FakeMicBackend:
    def __init__(self, tone_amplitude: int = 1000):
        self.tone_amplitude = tone_amplitude

    def record(self, duration_seconds: float, sample_rate: int, channels: int) -> bytes:
        n_frames = int(duration_seconds * sample_rate)
        return struct.pack(f"<{n_frames * channels}h", *([self.tone_amplitude] * n_frames * channels))


class FakeWhisperBackend:
    def __init__(self, text: str = "turn on the lights"):
        self.text = text

    def transcribe(self, samples: bytes, *, sample_rate: int, channels: int, model_size: str) -> dict:
        return {"text": self.text, "language": "en", "segments": [self.text]}


class FakeTTSBackend:
    def __init__(self, audio: bytes = b"fake-audio-bytes"):
        self.audio = audio

    def synthesize(self, text: str, *, voice: str | None) -> bytes:
        return self.audio


class ScriptedPlanner:
    """Fake `Planner` (mirrors `FakeWhisperBackend`/`FakeTTSBackend`): returns
    each step in `steps` in order, one per `plan()` call, regardless of the
    arguments -- enough to drive the worker's tool-call loop deterministically
    in tests without a real LLM."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.seen_history = []

    def plan(self, user_text, tool_results, history=()):
        self.seen_history.append(history)
        return self._steps.pop(0)


class AlwaysToolCallPlanner:
    """Fake `Planner` that never produces a final response -- used to prove
    the tool-call loop terminates deliberately (`ToolLoopExceeded`) instead
    of looping forever."""

    def plan(self, user_text, tool_results, history=()):
        return ToolCallStep(calls=(ToolCall(name="noop", arguments={}),))


def _dry_run_config():
    return LiveKitConfig.from_env(env={})


def test_build_room_client_returns_fake_in_dry_run():
    assert isinstance(build_room_client(_dry_run_config()), FakeRoomClient)


def test_worker_logs_connect_and_disconnect_lifecycle():
    worker = AgentWorker(_dry_run_config())
    asyncio.run(worker.run(max_runtime=0.01))
    names = [e for e, _ in worker.events]
    assert names[0] == EVENT_CONNECTED
    assert names[-1] == EVENT_DISCONNECTED


def test_worker_records_participant_join(caplog):
    config = _dry_run_config()
    room = FakeRoomClient(config)
    worker = AgentWorker(config, room=room)

    async def scenario():
        task = asyncio.create_task(worker.run())
        await asyncio.sleep(0)
        room.simulate_participant_join("test-client")
        assert room.participants == ["test-client"]
        worker.stop()
        await task

    with caplog.at_level("INFO"):
        asyncio.run(scenario())

    assert (EVENT_PARTICIPANT_JOINED, {"identity": "test-client"}) in worker.events
    assert "participant joined: test-client" in caplog.text


def test_worker_stop_makes_run_return():
    worker = AgentWorker(_dry_run_config())

    async def scenario():
        task = asyncio.create_task(worker.run())
        worker.stop()
        await asyncio.wait_for(task, timeout=1.0)

    asyncio.run(scenario())
    assert worker.room.connected is False


def test_max_runtime_terminates_without_stop():
    worker = AgentWorker(_dry_run_config())
    asyncio.run(asyncio.wait_for(worker.run(max_runtime=0.05), timeout=1.0))
    assert worker.room.connected is False


def test_capture_local_audio_without_source_raises():
    worker = AgentWorker(_dry_run_config())
    with pytest.raises(RuntimeError, match="No audio_source configured"):
        worker.capture_local_audio()


def test_capture_local_audio_uses_voice_core_abstraction():
    """Core acceptance criterion: the real-time pipeline captures local
    audio through voice_core's shared AudioSource/AudioBuffer, not a
    separate LiveKit-specific capture path."""
    source = MicAudioSource(FakeMicBackend(), duration_seconds=0.5, sample_rate=16_000, channels=1)
    worker = AgentWorker(_dry_run_config(), audio_source=source)

    buf = worker.capture_local_audio()

    assert isinstance(buf, AudioBuffer)
    assert worker.local_audio is buf
    assert buf.sample_rate == 16_000
    assert buf.duration_seconds == pytest.approx(0.5)


def test_run_captures_local_audio_when_source_configured():
    source = MicAudioSource(FakeMicBackend(), duration_seconds=0.01, sample_rate=16_000, channels=1)
    worker = AgentWorker(_dry_run_config(), audio_source=source)

    asyncio.run(worker.run(max_runtime=0.01))

    assert worker.local_audio is not None
    assert isinstance(worker.local_audio, AudioBuffer)


def test_transcribe_local_audio_without_stt_raises():
    source = MicAudioSource(FakeMicBackend(), duration_seconds=0.01, sample_rate=16_000, channels=1)
    worker = AgentWorker(_dry_run_config(), audio_source=source)
    worker.capture_local_audio()

    with pytest.raises(RuntimeError, match="No stt configured"):
        worker.transcribe_local_audio("run-1")


def test_transcribe_local_audio_without_capture_raises():
    worker = AgentWorker(_dry_run_config(), stt=SpeechToText(FakeWhisperBackend()))

    with pytest.raises(RuntimeError, match="No local audio captured"):
        worker.transcribe_local_audio("run-1")


def test_transcribe_local_audio_goes_through_voice_core_stt_and_records_latency():
    """Core acceptance criterion: streaming STT goes through the shared
    voice_core Whisper wrapper, and the stage is recorded via the Phase 1
    LatencyTracker."""
    source = MicAudioSource(FakeMicBackend(), duration_seconds=0.5, sample_rate=16_000, channels=1)
    stt = SpeechToText(FakeWhisperBackend(text="turn on the lights"), model_size="small")
    worker = AgentWorker(_dry_run_config(), audio_source=source, stt=stt)
    worker.capture_local_audio()

    result = worker.transcribe_local_audio("run-1")

    assert result.text == "turn on the lights"
    assert worker.last_transcript is result
    run = worker.latency.get_run("run-1")
    assert run.get("stt") is not None


def test_speak_without_tts_raises():
    worker = AgentWorker(_dry_run_config())

    with pytest.raises(RuntimeError, match="No tts configured"):
        worker.speak("hello", "out.mp3", "run-1")


def test_speak_goes_through_voice_core_tts_and_records_latency(tmp_path):
    """Core acceptance criterion: TTS output goes through the shared
    voice_core TTS wrapper, and the stage is recorded via the Phase 1
    LatencyTracker."""
    tts = TextToSpeech(FakeTTSBackend(audio=b"some audio"), provider="pyttsx3", voice="default")
    worker = AgentWorker(_dry_run_config(), tts=tts)
    out = tmp_path / "reply.mp3"

    result = worker.speak("hello there", out, "run-1")

    assert result.audio_path == str(out)
    assert out.read_bytes() == b"some audio"
    run = worker.latency.get_run("run-1")
    assert run.get("tts") is not None


def test_transcribe_and_speak_accumulate_on_the_same_run(tmp_path):
    source = MicAudioSource(FakeMicBackend(), duration_seconds=0.01, sample_rate=16_000, channels=1)
    stt = SpeechToText(FakeWhisperBackend())
    tts = TextToSpeech(FakeTTSBackend())
    worker = AgentWorker(_dry_run_config(), audio_source=source, stt=stt, tts=tts)
    worker.capture_local_audio()

    worker.transcribe_local_audio("run-1")
    worker.speak("ok", tmp_path / "reply.mp3", "run-1")

    run = worker.latency.get_run("run-1")
    assert {s.stage for s in run.stages} == {"stt", "tts"}


# --- tool/function calling (issue #7) ---------------------------------


def test_handle_user_turn_without_planner_raises():
    worker = AgentWorker(_dry_run_config())

    with pytest.raises(RuntimeError, match="No planner configured"):
        asyncio.run(worker.handle_user_turn("turn on the lights", "run-1"))


def test_handle_user_turn_executes_tool_call_then_returns_final_response():
    tools = ToolRegistry()
    calls = []
    tools.register("turn_on_lights", lambda room: calls.append(room) or "ok")
    planner = ScriptedPlanner(
        [
            ToolCallStep(calls=(ToolCall(name="turn_on_lights", arguments={"room": "kitchen"}),)),
            FinalResponse(text="I've turned on the kitchen lights."),
        ]
    )
    worker = AgentWorker(_dry_run_config(), tools=tools, planner=planner)

    response = asyncio.run(worker.handle_user_turn("turn on the kitchen lights", "run-1"))

    assert response == "I've turned on the kitchen lights."
    assert calls == ["kitchen"]
    assert worker.state is RunState.IDLE
    assert (EVENT_TOOL_CALL, {"run_id": "run-1", "tool": "turn_on_lights", "arguments": {"room": "kitchen"}}) in worker.events
    run = worker.latency.get_run("run-1")
    assert [s.stage for s in run.stages] == ["llm", "llm"]


def test_handle_user_turn_feeds_unknown_tool_error_back_to_planner():
    planner = ScriptedPlanner(
        [
            ToolCallStep(calls=(ToolCall(name="does_not_exist", arguments={}),)),
            FinalResponse(text="Sorry, I can't do that."),
        ]
    )
    worker = AgentWorker(_dry_run_config(), planner=planner)

    response = asyncio.run(worker.handle_user_turn("do the impossible thing", "run-1"))

    assert response == "Sorry, I can't do that."


def test_handle_user_turn_raises_tool_loop_exceeded_when_planner_never_finishes():
    worker = AgentWorker(_dry_run_config(), planner=AlwaysToolCallPlanner())
    worker.tools.register("noop", lambda: None)

    with pytest.raises(ToolLoopExceeded):
        asyncio.run(worker.handle_user_turn("loop forever", "run-1"))

    assert worker.state is RunState.IDLE


# --- conversation memory (issue #8) -------------------------------------


def test_handle_user_turn_records_completed_turn_into_memory():
    planner = ScriptedPlanner([FinalResponse(text="Sure, done.")])
    worker = AgentWorker(_dry_run_config(), planner=planner)

    asyncio.run(worker.handle_user_turn("turn on the lights", "run-1", session_id="caller-a"))

    assert worker.memory.history("caller-a") == (
        Turn(user_text="turn on the lights", agent_text="Sure, done."),
    )


def test_handle_user_turn_passes_prior_turns_to_planner_as_history():
    planner = ScriptedPlanner([FinalResponse(text="It's sunny.")])
    worker = AgentWorker(_dry_run_config(), planner=planner)
    worker.memory.add_turn("caller-a", "hi", "hello there")

    asyncio.run(worker.handle_user_turn("what's the weather?", "run-2", session_id="caller-a"))

    assert planner.seen_history == [(Turn(user_text="hi", agent_text="hello there"),)]


def test_handle_user_turn_memory_persists_across_turns_in_the_same_session():
    planner = ScriptedPlanner(
        [FinalResponse(text="Nice to meet you, Sam."), FinalResponse(text="Your name is Sam.")]
    )
    worker = AgentWorker(_dry_run_config(), planner=planner)

    asyncio.run(worker.handle_user_turn("my name is Sam", "run-1", session_id="caller-a"))
    asyncio.run(worker.handle_user_turn("what's my name?", "run-2", session_id="caller-a"))

    history = worker.memory.history("caller-a")
    assert [t.user_text for t in history] == ["my name is Sam", "what's my name?"]
    assert planner.seen_history[1] == (Turn(user_text="my name is Sam", agent_text="Nice to meet you, Sam."),)


def test_handle_user_turn_memory_is_scoped_per_session():
    planner = ScriptedPlanner([FinalResponse(text="ok caller-a"), FinalResponse(text="ok caller-b")])
    worker = AgentWorker(_dry_run_config(), planner=planner)

    asyncio.run(worker.handle_user_turn("hello from a", "run-1", session_id="caller-a"))
    asyncio.run(worker.handle_user_turn("hello from b", "run-2", session_id="caller-b"))

    assert [t.user_text for t in worker.memory.history("caller-a")] == ["hello from a"]
    assert [t.user_text for t in worker.memory.history("caller-b")] == ["hello from b"]


def test_handle_user_turn_defaults_session_id_to_run_id():
    planner = ScriptedPlanner([FinalResponse(text="ok")])
    worker = AgentWorker(_dry_run_config(), planner=planner)

    asyncio.run(worker.handle_user_turn("hello", "run-1"))

    assert [t.user_text for t in worker.memory.history("run-1")] == ["hello"]


def test_handle_user_turn_abandoned_by_tool_loop_exceeded_is_not_recorded():
    worker = AgentWorker(_dry_run_config(), planner=AlwaysToolCallPlanner())
    worker.tools.register("noop", lambda: None)

    with pytest.raises(ToolLoopExceeded):
        asyncio.run(worker.handle_user_turn("loop forever", "run-1", session_id="caller-a"))

    assert worker.memory.history("caller-a") == ()


def test_conversation_memory_is_bounded_within_a_very_long_session():
    memory = ConversationMemory(max_turns=3)
    for i in range(10):
        memory.add_turn("caller-a", f"message {i}", f"reply {i}")

    history = memory.history("caller-a")

    assert len(history) == 3
    assert [t.user_text for t in history] == ["message 7", "message 8", "message 9"]


# --- barge-in (issue #7) ------------------------------------------------


def test_handle_barge_in_is_a_noop_when_not_speaking():
    worker = AgentWorker(_dry_run_config())

    interrupted = asyncio.run(worker.handle_barge_in())

    assert interrupted is False


def test_begin_speaking_then_barge_in_interrupts_and_records_event(tmp_path):
    tts = TextToSpeech(FakeTTSBackend())
    worker = AgentWorker(_dry_run_config(), tts=tts)

    async def scenario():
        task = worker.begin_speaking(
            "this is a long response the user will interrupt",
            tmp_path / "reply.mp3",
            "run-1",
            playback_seconds=5.0,
        )
        await asyncio.sleep(0)
        assert worker.state is RunState.SPEAKING
        interrupted = await worker.handle_barge_in()
        assert task.cancelled()
        return interrupted

    interrupted = asyncio.run(scenario())

    assert interrupted is True
    assert worker.state is RunState.LISTENING
    assert any(event == EVENT_INTERRUPTED and data["run_id"] == "run-1" for event, data in worker.events)


def test_speak_and_play_returns_to_idle_when_playback_finishes_uninterrupted(tmp_path):
    tts = TextToSpeech(FakeTTSBackend())
    worker = AgentWorker(_dry_run_config(), tts=tts)

    result = asyncio.run(
        worker.speak_and_play("hi", tmp_path / "reply.mp3", "run-1", playback_seconds=0.0)
    )

    assert result.audio_path == str(tmp_path / "reply.mp3")
    assert worker.state is RunState.IDLE
