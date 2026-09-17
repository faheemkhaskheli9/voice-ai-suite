"""LiveKit access-token minting.

Ported from realtime-voice-agent's src/rtva/token.py. LiveKit access tokens
are JWTs signed with the project API secret, carrying a ``video`` grant
claim. This is enough for a test client (or the agent worker) to join a
room. Kept dependency-light (PyJWT only) rather than pulling the full
server SDK.
"""

from __future__ import annotations

import time

import jwt


def create_access_token(
    api_key: str,
    api_secret: str,
    *,
    identity: str,
    room: str,
    ttl_seconds: int = 3600,
    can_publish: bool = True,
    can_subscribe: bool = True,
) -> str:
    if not api_key or not api_secret:
        raise ValueError("api_key and api_secret are required")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")

    now = int(time.time())
    claims = {
        "iss": api_key,
        "sub": identity,
        "nbf": now,
        "iat": now,
        "exp": now + ttl_seconds,
        "jti": identity,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": can_publish,
            "canSubscribe": can_subscribe,
        },
    }
    return jwt.encode(claims, api_secret, algorithm="HS256")


def decode_access_token(token: str, api_secret: str) -> dict:
    """Verify + decode (used by tests and debugging)."""

    return jwt.decode(token, api_secret, algorithms=["HS256"])
