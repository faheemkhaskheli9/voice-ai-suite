import asyncio
import struct

import pytest

from realtime_agent.config import LiveKitConfig
from realtime_agent.room import (
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_PARTICIPANT_JOINED,
    FakeRoomClient,
    build_room_client,
)
from realtime_agent.worker import AgentWorker
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
