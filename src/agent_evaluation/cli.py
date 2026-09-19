"""Persona CLI: ``python -m agent_evaluation.cli validate PATH`` / ``show PATH``.

``PATH`` may be a single ``.yaml`` file or a directory of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .persona import PersonaError, load_persona, load_personas


def _collect(path: Path):
    if path.is_dir():
        return load_personas(path)
    return [load_persona(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_evaluation", description="Validate/inspect persona scripts"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "show"):
        sp = sub.add_parser(name)
        sp.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    try:
        personas = _collect(args.path)
    except PersonaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.command == "validate":
        for p in personas:
            print(f"OK  {p.name}  ({len(p.sample_utterances)} utterances, "
                  f"{len(p.success_criteria)} success criteria)")
        print(f"{len(personas)} persona(s) valid")
        return 0

    for p in personas:
        print(json.dumps(p.model_dump(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
