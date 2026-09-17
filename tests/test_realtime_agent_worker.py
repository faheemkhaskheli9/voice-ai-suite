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


class FakeMicBackend:
    def __init__(self, tone_amplitude: int = 1000):
        self.tone_amplitude = tone_amplitude

    def record(self, duration_seconds: float, sample_rate: int, channels: int) -> bytes:
        n_frames = int(duration_seconds * sample_rate)
        return struct.pack(f"<{n_frames * channels}h", *([self.tone_amplitude] * n_frames * channels))


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
