"""Agent worker: connect to a room, log lifecycle events, run until stopped.

Ported from realtime-voice-agent's src/rtva/worker.py. Deliberate
termination (knowledge-base pattern "Stateful agentic run lifecycle"): the
worker never loops unbounded — it runs until ``stop()`` is called, an
optional ``max_runtime`` elapses, or ``connect()`` fails, and it always
disconnects cleanly on the way out.

Local audio capture (when an ``audio_source`` is supplied) goes through
``voice_core.audio``'s shared ``AudioSource``/``AudioBuffer`` abstraction
rather than a separate LiveKit-specific capture path — the same
mic/file-source interface every other feature app in this suite already
uses (README.md Section 2/4).

Issue #6: streaming STT/TTS also go through the shared ``voice_core``
wrappers (``SpeechToText``/``TextToSpeech``) rather than each feature app
wiring Whisper/a TTS provider itself, and every stage is timed through the
Phase 1 ``LatencyTracker`` so the real-time agent's per-turn timings are
retrievable the same way as every other feature app's.

Issue #7: barge-in (interrupting the agent mid-response) and tool/function
calling mid-conversation, both following the knowledge-base "Stateful
agentic run lifecycle" pattern -- see ``dialogue.py`` for the bounded
tool-call loop and the ``RunState`` taxonomy. "Speaking" playback has no
real audio-output hardware to drive in this environment (or CI), so it is
simulated as an interruptible sleep sized to the response text
(``_estimate_playback_seconds``) rather than actually playing audio.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from voice_core.audio import AudioBuffer, AudioSource
from voice_core.latency import LatencyTracker
from voice_core.stt import SpeechToText, TranscriptionResult
from voice_core.tts import SynthesisResult, TextToSpeech

from .config import LiveKitConfig
from .dialogue import (
    MAX_TOOL_ITERATIONS,
    FinalResponse,
    Planner,
    RunState,
    ToolCall,
    ToolCallStep,
    ToolLoopExceeded,
    ToolRegistry,
    ToolResult,
)
from .room import (
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_PARTICIPANT_JOINED,
    EVENT_PARTICIPANT_LEFT,
    RoomClient,
    build_room_client,
)

logger = logging.getLogger("realtime_agent.worker")

EVENT_INTERRUPTED = "interrupted"
EVENT_TOOL_CALL = "tool_call"

#: Rough spoken-word rate used only to size the mocked "speaking" wait in
#: `speak_and_play` below -- not a real TTS/playback timing.
_WORDS_PER_SECOND = 2.5


def _estimate_playback_seconds(text: str) -> float:
    word_count = len(text.split())
    return max(word_count / _WORDS_PER_SECOND, 0.01)


@dataclass
class AgentWorker:
    config: LiveKitConfig
    room: RoomClient = field(default=None)  # type: ignore[assignment]
    audio_source: AudioSource | None = None
    stt: SpeechToText | None = None
    tts: TextToSpeech | None = None
    latency: LatencyTracker = field(default_factory=LatencyTracker)
    tools: ToolRegistry = field(default_factory=ToolRegistry)
    planner: Planner | None = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    events: list[tuple[str, dict]] = field(default_factory=list)
    local_audio: AudioBuffer | None = field(default=None, repr=False)
    last_transcript: TranscriptionResult | None = field(default=None, repr=False)
    _state: RunState = field(default=RunState.IDLE, repr=False)
    _speaking_task: "asyncio.Task[SynthesisResult] | None" = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.room is None:
            self.room = build_room_client(self.config)
        self.room.on_event(self._record_event)

    @property
    def state(self) -> RunState:
        return self._state

    # --- lifecycle logging --------------------------------------------

    def _record_event(self, event: str, data: dict) -> None:
        self.events.append((event, data))
        if event == EVENT_CONNECTED:
            logger.info("connected to room %s", data.get("room"))
        elif event == EVENT_DISCONNECTED:
            logger.info("disconnected from room %s", data.get("room"))
        elif event == EVENT_PARTICIPANT_JOINED:
            logger.info("participant joined: %s", data.get("identity"))
        elif event == EVENT_PARTICIPANT_LEFT:
            logger.info("participant left: %s", data.get("identity"))
        elif event == EVENT_INTERRUPTED:
            logger.info("speech interrupted (barge-in) for run %s", data.get("run_id"))
        elif event == EVENT_TOOL_CALL:
            logger.info("tool call for run %s: %s(%s)", data.get("run_id"), data.get("tool"), data.get("arguments"))

    # --- audio -----------------------------------------------------------

    def capture_local_audio(self) -> AudioBuffer:
        """Capture one clip via the shared `voice_core.audio` abstraction
        (`MicAudioSource`/`FileAudioSource`) instead of a separate,
        LiveKit-specific capture path. Raises if no `audio_source` was
        configured, rather than silently skipping capture."""

        if self.audio_source is None:
            raise RuntimeError(
                "No audio_source configured; pass a voice_core.audio.AudioSource "
                "(e.g. MicAudioSource or FileAudioSource) to capture local audio."
            )
        self.local_audio = self.audio_source.capture()
        return self.local_audio

    # --- streaming STT/TTS -------------------------------------------

    def transcribe_local_audio(self, run_id: str) -> TranscriptionResult:
        """Transcribe the last-captured local audio through the shared
        ``voice_core`` Whisper wrapper, timed under `run_id`'s "stt" stage
        so the turn's latency is retrievable the same way as every other
        feature app's. Raises if no `stt` wrapper was configured or no
        audio has been captured yet, rather than silently skipping."""

        if self.stt is None:
            raise RuntimeError(
                "No stt configured; pass a voice_core.stt.SpeechToText to transcribe audio."
            )
        if self.local_audio is None:
            raise RuntimeError("No local audio captured yet; call capture_local_audio() first.")

        with self.latency.stage(run_id, "stt"):
            result = self.stt.transcribe(self.local_audio)
        self.last_transcript = result
        logger.info("transcribed local audio (run %s): %r", run_id, result.text)
        return result

    def speak(self, text: str, output_path: str | Path, run_id: str) -> SynthesisResult:
        """Synthesize `text` through the shared ``voice_core`` TTS wrapper,
        timed under `run_id`'s "tts" stage. Raises if no `tts` wrapper was
        configured, rather than silently skipping output."""

        if self.tts is None:
            raise RuntimeError(
                "No tts configured; pass a voice_core.tts.TextToSpeech to synthesize audio."
            )

        with self.latency.stage(run_id, "tts"):
            result = self.tts.synthesize(text, output_path)
        logger.info("synthesized TTS output (run %s) -> %s", run_id, result.audio_path)
        return result

    # --- speaking / barge-in ------------------------------------------

    async def speak_and_play(
        self,
        text: str,
        output_path: str | Path,
        run_id: str,
        playback_seconds: float | None = None,
    ) -> SynthesisResult:
        """Synthesize `text` (via `speak`) then simulate playback for
        `playback_seconds` (default: `_estimate_playback_seconds(text)`) --
        long enough for `handle_barge_in` to interrupt it before it
        finishes, which is what makes barge-in observable without real
        audio-output hardware. Raises `asyncio.CancelledError` (recording an
        `EVENT_INTERRUPTED` event first) if cancelled mid-playback."""

        result = self.speak(text, output_path, run_id)
        wait = _estimate_playback_seconds(text) if playback_seconds is None else playback_seconds
        self._state = RunState.SPEAKING
        try:
            await asyncio.sleep(wait)
        except asyncio.CancelledError:
            self._record_event(EVENT_INTERRUPTED, {"run_id": run_id, "text": text})
            raise
        else:
            self._state = RunState.IDLE
        return result

    def begin_speaking(
        self,
        text: str,
        output_path: str | Path,
        run_id: str,
        playback_seconds: float | None = None,
    ) -> "asyncio.Task[SynthesisResult]":
        """Start `speak_and_play` as a cancellable task and remember it so a
        later `handle_barge_in` call can interrupt it."""

        self._speaking_task = asyncio.create_task(
            self.speak_and_play(text, output_path, run_id, playback_seconds)
        )
        return self._speaking_task

    async def handle_barge_in(self) -> bool:
        """Interrupt in-progress speech playback because the user started
        talking over the agent. Returns True if a response was actually
        interrupted, False if the agent wasn't speaking (nothing to
        interrupt) -- so a caller can tell a real barge-in apart from a
        no-op."""

        if self._state is not RunState.SPEAKING or self._speaking_task is None:
            return False
        task = self._speaking_task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._state = RunState.LISTENING
        return True

    # --- tool/function calling -----------------------------------------

    async def handle_user_turn(self, user_text: str, run_id: str) -> str:
        """Run one bounded plan -> tool-call -> resume cycle for a user's
        utterance (stateful-run-lifecycle KB pattern) and return the final
        response text.

        Each iteration asks `planner` whether to call tools or respond;
        tool calls execute and their results (including failures, fed back
        as an error rather than raised) are given to the planner on the
        next iteration. The loop always terminates: either the planner
        returns a `FinalResponse`, or `MAX_TOOL_ITERATIONS` is reached and
        `ToolLoopExceeded` is raised -- never an unbounded tool-call chain.
        """

        if self.planner is None:
            raise RuntimeError(
                "No planner configured; pass a dialogue.Planner to decide "
                "responses/tool calls for this turn."
            )

        self._state = RunState.PROCESSING
        tool_results: list[ToolResult] = []
        for _ in range(MAX_TOOL_ITERATIONS):
            with self.latency.stage(run_id, "llm"):
                step = self.planner.plan(user_text, tuple(tool_results))
            if isinstance(step, FinalResponse):
                self._state = RunState.IDLE
                return step.text
            assert isinstance(step, ToolCallStep)
            for call in step.calls:
                tool_results.append(self._run_tool_call(call, run_id))

        self._state = RunState.IDLE
        raise ToolLoopExceeded(run_id, MAX_TOOL_ITERATIONS)

    def _run_tool_call(self, call: ToolCall, run_id: str) -> ToolResult:
        self._record_event(
            EVENT_TOOL_CALL, {"run_id": run_id, "tool": call.name, "arguments": call.arguments}
        )
        try:
            fn = self.tools.get(call.name)
            output = fn(**call.arguments)
        except Exception as exc:  # noqa: BLE001 - fed back to the planner as an error, not a crash
            return ToolResult(call=call, error=str(exc))
        return ToolResult(call=call, output=output)

    # --- run loop ----------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    async def run(self, max_runtime: float | None = None) -> None:
        mode = "connect" if self.config.is_configured else "dry-run"
        logger.info("starting agent worker (%s mode) as %r", mode, self.config.agent_identity)
        await self.room.connect()
        try:
            if self.audio_source is not None:
                self.capture_local_audio()
            waiter = asyncio.create_task(self._stop.wait())
            done, pending = await asyncio.wait({waiter}, timeout=max_runtime)
            for task in pending:
                task.cancel()
        finally:
            await self.room.disconnect()
            logger.info("agent worker stopped")

    async def run_until_idle(self) -> None:
        """Connect, drain any already-queued events, disconnect. Used by smoke tests."""

        await self.run(max_runtime=0.01)
