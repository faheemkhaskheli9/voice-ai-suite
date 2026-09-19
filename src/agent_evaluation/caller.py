"""Simulated caller: drives a voice session from a persona script.

Real LiveKit needs a running server + network + (for real speech) a paid
TTS/STT provider, none of which this repo's CPU-only dev/CI convention has.
The turn-taking loop itself -- connect, speak each scripted utterance, wait
for the agent's spoken response before advancing -- is the thing under test
here, so it is written against a small ``VoiceRoomBackend`` protocol. A
``FakeVoiceRoomBackend`` (in-memory, no network) exercises the loop in tests
and against a stub agent; the real backend lazily imports ``livekit`` only
when actually used, raising :class:`LiveKitUnavailableError` with an
actionable message instead of a bare ``ImportError`` if it isn't installed.

Loop shape follows the portfolio's stateful-run-lifecycle knowledge-base
pattern (E:\\Projects\\LLM\\knowledge-base\\agentic-loops\\stateful-run-lifecycle.md):
a hard turn cap (the shorter of the script length and the persona's own
``max_turns``), an explicit branch on each turn's outcome (agent responded /
timed out / room closed) rather than collapsing every non-response into one
generic failure, and a single terminal :class:`CallStatus` recorded at the
end instead of relying on the model/agent to self-report "done".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from .persona import Persona

__all__ = [
    "CallStatus",
    "CallTurn",
    "CallResult",
    "LiveKitUnavailableError",
    "RoomClosedError",
    "VoiceRoomBackend",
    "FakeVoiceRoomBackend",
    "LiveKitRoomBackend",
    "run_persona_call",
]


class LiveKitUnavailableError(RuntimeError):
    """Raised when a real LiveKit session is requested but impossible here.

    Covers: the ``livekit`` package not installed, and no reachable LiveKit
    server URL configured -- both expected on this repo's CPU-only dev/CI
    hosts with no external services provisioned.
    """


class RoomClosedError(RuntimeError):
    """The voice room disconnected mid-call (agent crashed, server dropped it)."""


class CallStatus(str, Enum):
    SCRIPT_COMPLETE = "script_complete"  # ran every scripted utterance
    MAX_TURNS_REACHED = "max_turns_reached"  # hit persona.max_turns before the script ended
    TIMEOUT = "timeout"  # agent never responded to a turn within the budget
    ROOM_CLOSED = "room_closed"  # room/connection dropped mid-call
    ERROR = "error"  # backend raised something else


@dataclass(frozen=True)
class CallTurn:
    turn_index: int
    caller_utterance: str
    agent_response: str | None  # None only when the turn's status isn't "responded"


@dataclass(frozen=True)
class CallResult:
    persona_name: str
    room_name: str
    status: CallStatus
    turns: tuple[CallTurn, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def completed(self) -> bool:
        return self.status in (CallStatus.SCRIPT_COMPLETE, CallStatus.MAX_TURNS_REACHED)


class VoiceRoomBackend(Protocol):
    """Anything that can carry a persona-driven call over a voice room.

    "Audio" here is represented as plain text (the utterance/response text
    that would be synthesized/transcribed); no real TTS/STT is required to
    exercise the turn-taking loop, and a real backend is free to do the
    synthesis/transcription itself before returning.
    """

    def connect(self, room_name: str) -> None: ...

    def publish_utterance(self, text: str) -> None: ...

    def await_response(self, timeout: float) -> str | None:
        """Return the agent's response text, or None on timeout."""
        ...

    def disconnect(self) -> None: ...


class FakeVoiceRoomBackend:
    """In-memory backend for tests and for exercising the loop against a stub agent.

    ``agent`` is any ``Callable[[str], str | None]`` -- the stub voice agent
    under test. Returning ``None`` simulates the agent never responding
    (a timeout); raising :class:`RoomClosedError` simulates a dropped call.
    """

    def __init__(self, agent) -> None:
        self._agent = agent
        self.connected = False
        self.room_name: str | None = None
        self.published: list[str] = []

    def connect(self, room_name: str) -> None:
        self.connected = True
        self.room_name = room_name

    def publish_utterance(self, text: str) -> None:
        if not self.connected:
            raise RoomClosedError("cannot publish before connecting")
        self.published.append(text)

    def await_response(self, timeout: float) -> str | None:
        if not self.connected:
            raise RoomClosedError("room is not connected")
        return self._agent(self.published[-1])

    def disconnect(self) -> None:
        self.connected = False


class LiveKitRoomBackend:
    """The production backend: a real LiveKit room, only usable with the SDK installed."""

    def __init__(self, url: str, token: str) -> None:
        self._url = url
        self._token = token
        self._room = None

    def connect(self, room_name: str) -> None:
        try:
            from livekit import rtc  # noqa: PLC0415 - only present when livekit is installed
        except ImportError as exc:
            raise LiveKitUnavailableError(
                "The 'livekit' package is not installed. A real voice session needs "
                "a running LiveKit server plus the `livekit` SDK; this repo's "
                "CPU-only dev/CI hosts have neither. Use FakeVoiceRoomBackend to "
                "exercise the caller loop, or install `livekit` and point --url/--token "
                "at a real deployment to run this against a live room."
            ) from exc

        self._room = rtc.Room()
        try:
            self._room.connect(self._url, self._token)
        except Exception as exc:  # the SDK raises assorted connection error types
            raise LiveKitUnavailableError(f"failed to connect to LiveKit room: {exc}") from exc

    def publish_utterance(self, text: str) -> None:
        raise LiveKitUnavailableError(
            "real audio publishing (TTS synthesis + track publish) is not implemented "
            "in this CPU-only build; use FakeVoiceRoomBackend for local runs and tests"
        )

    def await_response(self, timeout: float) -> str | None:
        raise LiveKitUnavailableError(
            "real audio capture (STT transcription of the agent's track) is not "
            "implemented in this CPU-only build; use FakeVoiceRoomBackend for local "
            "runs and tests"
        )

    def disconnect(self) -> None:
        if self._room is not None:
            self._room.disconnect()
            self._room = None


def run_persona_call(
    persona: Persona,
    backend: VoiceRoomBackend,
    room_name: str,
    *,
    turn_timeout: float = 10.0,
) -> CallResult:
    """Drive one end-to-end call: connect, run the persona's script, disconnect.

    Turn cap is ``min(len(persona.sample_utterances), persona.max_turns)`` --
    a hard bound regardless of how the script or backend behave. Each turn's
    outcome is branched on explicitly (responded / timed out / room closed)
    rather than folded into a single generic failure path.
    """
    script = persona.sample_utterances
    turn_cap = min(len(script), persona.max_turns)
    turns: list[CallTurn] = []

    try:
        backend.connect(room_name)
    except RoomClosedError as exc:
        return CallResult(
            persona_name=persona.name, room_name=room_name,
            status=CallStatus.ROOM_CLOSED, error=str(exc),
        )
    except Exception as exc:  # backend-specific failure (e.g. LiveKitUnavailableError)
        return CallResult(
            persona_name=persona.name, room_name=room_name,
            status=CallStatus.ERROR, error=str(exc),
        )

    try:
        for index in range(turn_cap):
            utterance = script[index]
            try:
                backend.publish_utterance(utterance)
                response = backend.await_response(turn_timeout)
            except RoomClosedError as exc:
                turns.append(CallTurn(index, utterance, agent_response=None))
                return CallResult(
                    persona_name=persona.name, room_name=room_name,
                    status=CallStatus.ROOM_CLOSED, turns=tuple(turns), error=str(exc),
                )

            if response is None:
                turns.append(CallTurn(index, utterance, agent_response=None))
                return CallResult(
                    persona_name=persona.name, room_name=room_name,
                    status=CallStatus.TIMEOUT, turns=tuple(turns),
                    error=f"no response to turn {index} within {turn_timeout}s",
                )

            turns.append(CallTurn(index, utterance, agent_response=response))

        status = (
            CallStatus.SCRIPT_COMPLETE
            if turn_cap == len(script)
            else CallStatus.MAX_TURNS_REACHED
        )
        return CallResult(
            persona_name=persona.name, room_name=room_name, status=status, turns=tuple(turns)
        )
    finally:
        backend.disconnect()
