"""Typed persona scripting format.

A persona describes how a simulated caller behaves so a QA engineer can
express realistic caller behaviour in YAML without writing code. The schema
is a pydantic model (``extra='forbid'`` so a mistyped key fails loudly rather
than being silently ignored).

``configs/persona_template.yaml`` is the annotated template and
``docs/persona_schema.md`` documents every field; keep all three in step.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class PersonaError(ValueError):
    """A persona file is missing, unparseable, or fails schema validation."""


class Persona(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, description="Short unique persona label")
    goal: str = Field(min_length=1, description="What the caller is trying to achieve")
    tone: str = Field(min_length=1, description="Caller demeanour, e.g. 'calm', 'frustrated'")
    sample_utterances: list[str] = Field(
        min_length=1, description="Example things the caller might say, in order"
    )
    success_criteria: list[str] = Field(
        min_length=1, description="Conditions that mark the call a pass"
    )
    max_turns: int = Field(default=12, ge=1, le=100, description="Hard cap on caller turns")
    language: str = Field(default="en", min_length=2, max_length=10)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("sample_utterances", "success_criteria")
    @classmethod
    def _no_blank_items(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("list items must be non-empty strings")
        return cleaned


def _format_validation_error(path: Path, exc: ValidationError) -> str:
    lines = [f"invalid persona {path.name}:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


def load_persona(path: str | Path) -> Persona:
    path = Path(path)
    if not path.is_file():
        raise PersonaError(f"persona file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PersonaError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise PersonaError(f"{path.name} must contain a YAML mapping at the top level")
    try:
        return Persona.model_validate(raw)
    except ValidationError as exc:
        raise PersonaError(_format_validation_error(path, exc)) from exc


def load_personas(directory: str | Path) -> list[Persona]:
    directory = Path(directory)
    if not directory.is_dir():
        raise PersonaError(f"persona directory not found: {directory}")
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".yaml", ".yml"})
    if not files:
        raise PersonaError(f"no .yaml persona files under {directory}")
    return [load_persona(p) for p in files]
