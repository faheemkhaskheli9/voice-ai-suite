"""Tests for the simulated caller.

Real LiveKit needs a running server and paid TTS/STT, so every test drives
the loop through :class:`FakeVoiceRoomBackend` against a stub voice agent --
the "single end-to-end call runs successfully against a stub/test agent"
acceptance criterion.
"""

from __future__ import annotations

import pytest

from agent_evaluation.caller import (
    CallStatus,
    FakeVoiceRoomBackend,
    LiveKitRoomBackend,
    LiveKitUnavailableError,
    RoomClosedError,
    run_persona_call,
)
from agent_evaluation.persona import Persona


def _persona(**overrides) -> Persona:
    defaults = dict(
        name="tester",
        goal="Get a refund.",
        tone="calm",
        sample_utterances=["Hello.", "I need a refund.", "Thanks."],
        success_criteria=["Agent processes the refund."],
    )
    defaults.update(overrides)
    return Persona(**defaults)


def _echo_agent(utterance: str) -> str:
    return f"heard: {utterance}"


def test_full_script_completes_successfully():
    persona = _persona()
    backend = FakeVoiceRoomBackend(_echo_agent)

    result = run_persona_call(persona, backend, "room-1")

    assert result.status == CallStatus.SCRIPT_COMPLETE
    assert result.completed
    assert len(result.turns) == 3
    assert result.turns[0].caller_utterance == "Hello."
    assert result.turns[0].agent_response == "heard: Hello."
    assert backend.connected is False  # disconnected on completion
    assert backend.room_name == "room-1"


def test_agent_hears_each_utterance_before_advancing():
    persona = _persona()
    backend = FakeVoiceRoomBackend(_echo_agent)

    run_persona_call(persona, backend, "room-1")

    assert backend.published == ["Hello.", "I need a refund.", "Thanks."]


def test_max_turns_caps_before_script_ends():
    persona = _persona(max_turns=2)
    backend = FakeVoiceRoomBackend(_echo_agent)

    result = run_persona_call(persona, backend, "room-1")

    assert result.status == CallStatus.MAX_TURNS_REACHED
    assert result.completed
    assert len(result.turns) == 2


def test_agent_timeout_stops_the_call():
    persona = _persona()
    backend = FakeVoiceRoomBackend(lambda utterance: None)  # never responds

    result = run_persona_call(persona, backend, "room-1")

    assert result.status == CallStatus.TIMEOUT
    assert not result.completed
    assert len(result.turns) == 1
    assert result.turns[0].agent_response is None
    assert "no response" in result.error


def test_dropped_room_mid_call_is_reported():
    calls = {"n": 0}

    def flaky_agent(utterance: str) -> str:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RoomClosedError("agent process crashed")
        return "ok"

    persona = _persona()
    backend = FakeVoiceRoomBackend(flaky_agent)

    result = run_persona_call(persona, backend, "room-1")

    assert result.status == CallStatus.ROOM_CLOSED
    # turn 0 succeeded before the drop; the failing turn is recorded with no response
    assert len(result.turns) == 2
    assert result.turns[0].agent_response == "ok"
    assert result.turns[1].agent_response is None


def test_connect_failure_is_reported_without_running_any_turns():
    class _RefusingBackend(FakeVoiceRoomBackend):
        def connect(self, room_name: str) -> None:
            raise RoomClosedError("server refused connection")

    persona = _persona()
    backend = _RefusingBackend(_echo_agent)

    result = run_persona_call(persona, backend, "room-1")

    assert result.status == CallStatus.ROOM_CLOSED
    assert result.turns == ()


def test_unexpected_backend_error_is_captured_not_raised():
    class _BoomBackend(FakeVoiceRoomBackend):
        def connect(self, room_name: str) -> None:
            raise ValueError("unexpected boom")

    persona = _persona()
    result = run_persona_call(persona, _BoomBackend(_echo_agent), "room-1")

    assert result.status == CallStatus.ERROR
    assert "boom" in result.error


def test_disconnect_is_called_even_on_timeout():
    persona = _persona()
    backend = FakeVoiceRoomBackend(lambda utterance: None)

    run_persona_call(persona, backend, "room-1")

    assert backend.connected is False


def test_real_backend_raises_clear_error_without_livekit_installed():
    try:
        import livekit  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("a real livekit install is present; the fallback path is not exercised")

    backend = LiveKitRoomBackend(url="wss://example.invalid", token="fake-token")
    with pytest.raises(LiveKitUnavailableError, match="livekit"):
        backend.connect("room-1")
