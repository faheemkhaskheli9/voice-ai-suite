"""Conversation memory for the Real-Time Voice Agent (issue #8).

Working-memory pattern (knowledge-base `memory/working-memory.md`): the
per-turn planner call (`dialogue.Planner.plan`) is stateless, so the app --
not the model -- owns the running history and re-injects it on every call.
History lives purely in-process for the worker's lifetime; if a session's
turns must outlive the process, that belongs in episodic memory, not here.

Turns are stored per `session_id` (not per `run_id`, which identifies a
single turn) so two callers' histories never mix even when one
`AgentWorker` serves several sessions concurrently (e.g. multiple LiveKit
rooms). Each session is capped at `max_turns`, oldest turn evicted first
(sliding-window truncation, the working-memory KB pattern's "keep only
recent N turns" strategy) so a long-running session's history never grows
unbounded.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

#: Sliding-window size: enough spoken turns to keep a conversation coherent
#: without letting an indefinitely long session grow the prompt unbounded.
DEFAULT_MAX_TURNS = 12


@dataclass(frozen=True)
class Turn:
    """One completed user/agent exchange, as recorded in conversation memory."""

    user_text: str
    agent_text: str


class ConversationMemory:
    """Per-session, bounded conversation history.

    `history(session_id)` returns oldest-first; `add_turn` appends and
    silently evicts the oldest turn once a session exceeds `max_turns`
    (via `deque(maxlen=...)`) rather than growing without bound.
    """

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns!r}")
        self._max_turns = max_turns
        self._sessions: dict[str, deque[Turn]] = {}

    def history(self, session_id: str) -> tuple[Turn, ...]:
        return tuple(self._sessions.get(session_id, ()))

    def add_turn(self, session_id: str, user_text: str, agent_text: str) -> None:
        session = self._sessions.setdefault(session_id, deque(maxlen=self._max_turns))
        session.append(Turn(user_text=user_text, agent_text=agent_text))

    def clear(self, session_id: str) -> None:
        """Drop a session's history entirely (e.g. when a call ends)."""
        self._sessions.pop(session_id, None)
