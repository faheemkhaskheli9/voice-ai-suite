"""LiveKit connection configuration for the Real-Time Voice Agent feature.

Ported from realtime-voice-agent's src/rtva/config.py. Loaded from the
environment (``.env`` in local dev). Two modes:

* **connect mode** — ``url`` + ``api_key`` + ``api_secret`` all present; the
  worker uses the real LiveKit client.
* **dry-run mode** — credentials absent; the worker runs against an in-process
  fake so the pipeline is demoable/testable with no server.

Rule 7 (portfolio robustness): a *partially* configured environment (e.g. a
URL but no secret) is a hard error, not a silent fall-through to dry-run —
that almost always means a misconfigured deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

_ENV_URL = "LIVEKIT_URL"
_ENV_KEY = "LIVEKIT_API_KEY"
_ENV_SECRET = "LIVEKIT_API_SECRET"
_DEFAULT_CONFIG_PATH = Path("configs/realtime_agent.yaml")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LiveKitConfig:
    url: str | None
    api_key: str | None
    api_secret: str | None
    room_name: str = "voice-agent-dev"
    agent_identity: str = "agent-worker"

    @property
    def is_configured(self) -> bool:
        return bool(self.url and self.api_key and self.api_secret)

    def require_configured(self) -> "LiveKitConfig":
        if not self.is_configured:
            raise ConfigError(
                "LiveKit credentials are incomplete; set "
                f"{_ENV_URL}, {_ENV_KEY}, {_ENV_SECRET}"
            )
        return self

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "LiveKitConfig":
        src = os.environ if env is None else env
        url = src.get(_ENV_URL) or None
        key = src.get(_ENV_KEY) or None
        secret = src.get(_ENV_SECRET) or None

        provided = [p for p in (url, key, secret) if p]
        if provided and len(provided) != 3:
            raise ConfigError(
                "partial LiveKit configuration: set all of "
                f"{_ENV_URL}, {_ENV_KEY}, {_ENV_SECRET} or none of them"
            )

        return cls(
            url=url,
            api_key=key,
            api_secret=secret,
            room_name=src.get("LIVEKIT_ROOM", "voice-agent-dev"),
            agent_identity=src.get("LIVEKIT_AGENT_IDENTITY", "agent-worker"),
        )

    @classmethod
    def from_yaml(
        cls, path: str | os.PathLike[str], env: dict[str, str] | None = None
    ) -> "LiveKitConfig":
        """Load non-secret settings (room name, agent identity) from a YAML
        file at `path`; secrets always come from the environment, never the
        config file, so a committed config never holds a credential.

        `path` given but missing is a hard error (rule 7: strict on
        explicit user input), matching `from_env`'s partial-credential
        error for a misconfigured deployment rather than a silent
        fall-through to defaults.
        """

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"config file not found: {file_path}")
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ConfigError("config file must contain a mapping at the top level")

        base = cls.from_env(env)
        room_name = raw.get("room_name", base.room_name)
        agent_identity = raw.get("agent_identity", base.agent_identity)
        return cls(
            url=base.url,
            api_key=base.api_key,
            api_secret=base.api_secret,
            room_name=str(room_name),
            agent_identity=str(agent_identity),
        )


def load_config(
    config_path: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
) -> LiveKitConfig:
    """Resolve the config path per rule 7: an explicit `config_path` that is
    missing is a hard error; omitted, fall back to `configs/realtime_agent.yaml`
    if present, else env vars alone."""

    if config_path is not None:
        return LiveKitConfig.from_yaml(config_path, env)
    if _DEFAULT_CONFIG_PATH.is_file():
        return LiveKitConfig.from_yaml(_DEFAULT_CONFIG_PATH, env)
    return LiveKitConfig.from_env(env)
