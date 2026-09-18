"""Tool/function-calling and turn-state abstractions for the Real-Time
Voice Agent (issue #7).

No real LLM API is available in this environment (or CI) -- no paid APIs
per the Definition of Done -- so deciding whether to call a tool or give a
final response goes through a pluggable `Planner` instead of an LLM
function-calling API directly. Production code supplies a real `Planner`
wrapping an LLM (OpenAI/Anthropic-style tool use); tests supply a fake one,
mirroring the `WhisperBackend`/`TTSBackend` pattern in `voice_core.stt`/
`voice_core.tts`.

The tool-call loop in `worker.AgentWorker.handle_user_turn` follows the
knowledge-base "Stateful agentic run lifecycle" pattern: each planner step
is either a tool-call request (pause the turn, execute, resume with
results) or a final response (terminal), and the loop is bounded by a hard
iteration cap (`MAX_TOOL_ITERATIONS`) rather than trusting the planner to
always terminate on its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

#: A registered tool: any callable taking keyword arguments and returning a
#: JSON-ish result. Real tools wrap e.g. a smart-home API, a calendar
#: lookup, etc.; tests register plain functions/lambdas.
ToolFunction = Callable[..., object]

#: Hard cap on planner<->tool round-trips for a single user turn (stateful-
#: run-lifecycle KB pattern: always a hard iteration cap sized to task
#: complexity -- a voice turn is a simple, single-intent exchange, so a
#: small cap is enough and keeps a misbehaving planner from looping forever).
MAX_TOOL_ITERATIONS = 5


class RunState(Enum):
    """Where a turn/the agent currently is -- the KB pattern's stop-reason
    taxonomy adapted to a voice turn instead of a hosted-agent run."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class ToolError(RuntimeError):
    """Base class for tool-registry/tool-loop problems."""


class ToolNotFound(ToolError):
    """Raised when a planner requests a tool that was never registered --
    carries the name and the known tools so the error is actionable."""

    def __init__(self, name: str, known: tuple[str, ...]):
        self.name = name
        super().__init__(
            f"No tool registered as {name!r}. Known tools: {', '.join(known) or '(none)'}."
        )


class ToolLoopExceeded(ToolError):
    """Raised when a turn hits `MAX_TOOL_ITERATIONS` without the planner
    producing a final response -- deliberate termination (stateful-run-
    lifecycle KB pattern) rather than looping forever."""

    def __init__(self, run_id: str, max_iterations: int):
        self.run_id = run_id
        self.max_iterations = max_iterations
        super().__init__(
            f"Run {run_id!r} exceeded {max_iterations} tool-call iteration(s) "
            "without a final response -- stopping deliberately rather than "
            "continuing an unbounded tool-call loop."
        )


@dataclass(frozen=True)
class ToolCall:
    """One tool/function-call request from the planner for this turn."""

    name: str
    arguments: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """One executed tool call's outcome, fed back to the planner on the
    next iteration so it can produce a final response (or retry)."""

    call: ToolCall
    output: object = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class ToolCallStep:
    """Planner step: pause the turn and execute these tool calls before
    resuming (the KB pattern's "requires_action" branch)."""

    calls: tuple[ToolCall, ...]


@dataclass(frozen=True)
class FinalResponse:
    """Planner step: the turn is done; speak this text (the KB pattern's
    terminal "completed" branch)."""

    text: str


PlannerStep = ToolCallStep | FinalResponse


class Planner(Protocol):
    """Decides, each iteration of a turn, whether to call tools or give a
    final response. Kept tiny and framework-free so it can be backed by a
    real LLM function-calling API or a test fake interchangeably."""

    def plan(self, user_text: str, tool_results: tuple[ToolResult, ...]) -> PlannerStep: ...


class ToolRegistry:
    """Name -> callable lookup for tools the planner can request mid-
    conversation. `get` raises `ToolNotFound` (rather than a bare
    `KeyError`) so a misbehaving planner's request is easy to diagnose."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(self, name: str, fn: ToolFunction) -> None:
        self._tools[name] = fn

    def get(self, name: str) -> ToolFunction:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFound(name, tuple(self._tools)) from None

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)
