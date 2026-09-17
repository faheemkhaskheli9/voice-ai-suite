"""Room transport abstraction.

Ported from realtime-voice-agent's src/rtva/room.py. The worker depends only
on :class:`RoomClient`. :class:`FakeRoomClient` (in-process, no server) is
used for dev/tests; the real LiveKit-backed client is created lazily by
:func:`build_room_client` so that ``import realtime_agent`` never requires
the ``livekit`` SDK to be installed (rule 12: import must be pure, cheap,
and total).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Protocol

from .config import LiveKitConfig

# lifecycle event names emitted to registered listeners
EVENT_CONNECTED = "connected"
EVENT_DISCONNECTED = "disconnected"
EVENT_PARTICIPANT_JOINED = "participant_joined"
EVENT_PARTICIPANT_LEFT = "participant_left"

Listener = Callable[[str, dict], None]


class RoomClient(Protocol):
    async def connect(self) -> None: ...
    async def disconnect(self) -> None: ...
    def on_event(self, listener: Listener) -> None: ...
    @property
    def participants(self) -> list[str]: ...
    @property
    def connected(self) -> bool: ...


@dataclass
class FakeRoomClient:
    """Deterministic in-process room. Tests drive participants directly."""

    config: LiveKitConfig
    _connected: bool = False
    _participants: list[str] = field(default_factory=list)
    _listeners: list[Listener] = field(default_factory=list)

    def on_event(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _emit(self, event: str, **data: object) -> None:
        for listener in self._listeners:
            listener(event, dict(data))

    async def connect(self) -> None:
        if self._connected:
            return
        await asyncio.sleep(0)
        self._connected = True
        self._emit(EVENT_CONNECTED, room=self.config.room_name)

    async def disconnect(self) -> None:
        if not self._connected:
            return
        self._connected = False
        self._emit(EVENT_DISCONNECTED, room=self.config.room_name)

    # --- test/dev helpers -------------------------------------------------

    def simulate_participant_join(self, identity: str) -> None:
        self._participants.append(identity)
        self._emit(EVENT_PARTICIPANT_JOINED, identity=identity)

    def simulate_participant_leave(self, identity: str) -> None:
        self._participants.remove(identity)
        self._emit(EVENT_PARTICIPANT_LEFT, identity=identity)

    @property
    def participants(self) -> list[str]:
        return list(self._participants)

    @property
    def connected(self) -> bool:
        return self._connected


def build_room_client(config: LiveKitConfig) -> RoomClient:
    """Return a real LiveKit client if configured, else the in-process fake."""

    if not config.is_configured:
        return FakeRoomClient(config)
    try:
        from .livekit_client import LiveKitRoomClient
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "LiveKit credentials are set but the 'livekit' package is not "
            "installed; run `pip install -r requirements.txt`"
        ) from exc
    return LiveKitRoomClient(config)  # pragma: no cover - needs a live server
