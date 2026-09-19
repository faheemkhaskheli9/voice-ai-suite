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

:class:`AgentUnderTestBackend` (issue #10) is a third ``VoiceRoomBackend``
that targets this suite's own Phase 2 real-time agent
(:class:`realtime_agent.worker.AgentWorker`) instead of an external LiveKit
deployment or a bare stub callable -- so the evaluation feature can exercise
the in-suite agent end to end, offline, the same way ``FakeVoiceRoomBackend``
exercises the turn-taking loop itself.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from .persona import Persona

if TYPE_CHECKING:
    from realtime_agent.worker import AgentWorker

__all__ = [
    "CallStatus",
    "CallTurn",
    "CallResult",
    "LiveKitUnavailableError",
    "RoomClosedError",
    "VoiceRoomBackend",
    "FakeVoiceRoomBackend",
    "LiveKitRoomBackend",
    "AgentUnderTestBackend",
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
    #: Which agent build/config this run exercised (e.g. "agent-worker@v1"),
    #: taken from the backend's ``target_label`` when it has one. ``None``
    #: for backends that don't identify a specific agent version.
    agent_target: str | None = None

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


class AgentUnderTestBackend:
    """Targets this suite's own Phase 2 real-time agent as the agent-under-test.

    Turns are driven through the worker's real ``handle_user_turn`` dialogue
    loop (planner + tool calling) rather than raw LiveKit audio -- consistent
    with :class:`VoiceRoomBackend`'s "audio is plain text" contract. The
    worker's own room client is whatever its config built (``FakeRoomClient``
    unless real LiveKit credentials are set -- see
    ``realtime_agent.room.build_room_client``), so this backend is
    offline-safe by default, the same CPU-only convention every other
    feature app in this suite follows.

    ``handle_user_turn`` is synchronous-planner-driven but declared
    ``async def``; each turn runs it on a dedicated worker thread (via
    ``asyncio.run`` inside a fresh event loop on that thread) so a genuinely
    slow/stuck planner is still bounded by ``timeout`` instead of blocking
    forever -- a plain ``asyncio.wait_for`` around a synchronous planner call
    would not preempt it, since nothing in the call ever yields to the loop.

    ``agent_version`` is recorded on ``target_label`` (surfaced on
    :class:`CallResult` as ``agent_target``) so an evaluation run can tell
    which build of the in-suite agent it exercised.
    """

    def __init__(self, worker: "AgentWorker", agent_version: str = "dev") -> None:
        if worker.planner is None:
            raise ValueError(
                "AgentUnderTestBackend requires an AgentWorker configured with a planner"
            )
        self.worker = worker
        self.target_label = f"{worker.config.agent_identity}@{agent_version}"
        self._session_id: str | None = None
        self._turn_index = 0
        self._pending_utterance: str | None = None

    def connect(self, room_name: str) -> None:
        self._session_id = room_name
        self._turn_index = 0
        try:
            asyncio.run(self.worker.room.connect())
        except Exception as exc:  # noqa: BLE001 - any transport failure means the room never opened
            raise RoomClosedError(f"agent-under-test failed to connect: {exc}") from exc

    def publish_utterance(self, text: str) -> None:
        if self._session_id is None:
            raise RoomClosedError("cannot publish before connecting")
        self._pending_utterance = text

    def await_response(self, timeout: float) -> str | None:
        if self._session_id is None or self._pending_utterance is None:
            raise RoomClosedError("cannot await a response before publishing an utterance")

        from realtime_agent.dialogue import ToolLoopExceeded  # noqa: PLC0415 - keeps this module importable without realtime_agent at collection time

        run_id = f"{self._session_id}-{self._turn_index}"
        self._turn_index += 1
        utterance = self._pending_utterance
        session_id = self._session_id

        def _run_turn() -> str:
            return asyncio.run(self.worker.handle_user_turn(utterance, run_id, session_id))

        pool = ThreadPoolExecutor(max_workers=1)
        future: Future[str] = pool.submit(_run_turn)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            return None
        except ToolLoopExceeded as exc:
            raise RoomClosedError(str(exc)) from exc
        finally:
            pool.shutdown(wait=False)

    def disconnect(self) -> None:
        if self._session_id is not None:
            asyncio.run(self.worker.room.disconnect())
        self._session_id = None


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
    #: Which agent build/config this run targets, if the backend identifies
    #: one (see AgentUnderTestBackend.target_label) -- carried onto every
    #: CallResult returned below so a report can tell runs apart by target.
    agent_target = getattr(backend, "target_label", None)

    try:
        backend.connect(room_name)
    except RoomClosedError as exc:
        return CallResult(
            persona_name=persona.name, room_name=room_name,
            status=CallStatus.ROOM_CLOSED, error=str(exc), agent_target=agent_target,
        )
    except Exception as exc:  # backend-specific failure (e.g. LiveKitUnavailableError)
        return CallResult(
            persona_name=persona.name, room_name=room_name,
            status=CallStatus.ERROR, error=str(exc), agent_target=agent_target,
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
                    agent_target=agent_target,
                )

            if response is None:
                turns.append(CallTurn(index, utterance, agent_response=None))
                return CallResult(
                    persona_name=persona.name, room_name=room_name,
                    status=CallStatus.TIMEOUT, turns=tuple(turns),
                    error=f"no response to turn {index} within {turn_timeout}s",
                    agent_target=agent_target,
                )

            turns.append(CallTurn(index, utterance, agent_response=response))

        status = (
            CallStatus.SCRIPT_COMPLETE
            if turn_cap == len(script)
            else CallStatus.MAX_TURNS_REACHED
        )
        return CallResult(
            persona_name=persona.name, room_name=room_name, status=status, turns=tuple(turns),
            agent_target=agent_target,
        )
    finally:
        backend.disconnect()
