"""Real LiveKit-backed :class:`~realtime_agent.room.RoomClient`.

Ported from realtime-voice-agent's src/rtva/livekit_client.py. Imported
lazily by :func:`realtime_agent.room.build_room_client` only when
credentials are present, so the ``livekit`` SDK stays an optional dependency
for anyone running the dry-run path.

This module is intentionally thin: it maps LiveKit room events onto the
generic lifecycle events the worker listens for. It is exercised against a
real LiveKit server, not in unit tests.
"""

from __future__ import annotations

from .config import LiveKitConfig
from .room import (
    EVENT_CONNECTED,
    EVENT_DISCONNECTED,
    EVENT_PARTICIPANT_JOINED,
    EVENT_PARTICIPANT_LEFT,
    Listener,
)
from .token import create_access_token


class LiveKitRoomClient:  # pragma: no cover - requires a live LiveKit server
    def __init__(self, config: LiveKitConfig) -> None:
        from livekit import rtc  # noqa: F401  (import-time check)

        self._config = config.require_configured()
        self._rtc = rtc
        self._room = rtc.Room()
        self._listeners: list[Listener] = []

        self._room.on("connected", lambda: self._emit(EVENT_CONNECTED, room=config.room_name))
        self._room.on("disconnected", lambda *_: self._emit(EVENT_DISCONNECTED, room=config.room_name))
        self._room.on(
            "participant_connected",
            lambda p: self._emit(EVENT_PARTICIPANT_JOINED, identity=p.identity),
        )
        self._room.on(
            "participant_disconnected",
            lambda p: self._emit(EVENT_PARTICIPANT_LEFT, identity=p.identity),
        )

    def on_event(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _emit(self, event: str, **data: object) -> None:
        for listener in self._listeners:
            listener(event, dict(data))

    async def connect(self) -> None:
        token = create_access_token(
            self._config.api_key,
            self._config.api_secret,
            identity=self._config.agent_identity,
            room=self._config.room_name,
        )
        await self._room.connect(self._config.url, token)

    async def disconnect(self) -> None:
        await self._room.disconnect()

    @property
    def participants(self) -> list[str]:
        return [p.identity for p in self._room.remote_participants.values()]

    @property
    def connected(self) -> bool:
        return self._room.connection_state == self._rtc.ConnectionState.CONN_CONNECTED
