"""Dataset assembly CLI: ``python -m speech_evaluator.cli --audio-dir DIR --out DIR``.

Only assembles the manifest -- running the actual size/version comparison
needs a real `WhisperBackend` (see `comparison.py`'s module docstring), which
this scaffold-stage environment doesn't have, so `run_comparison` is a
library call for now rather than a CLI subcommand (mirrors the archived
`speech-model-evaluator` repo's own `sme/cli.py`, which was assembly-only).
"""
from __future__ import annotations

import argparse
import sys

from .dataset import DatasetError, assemble_dataset, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speech-evaluator-assemble",
        description="Assemble a reference audio/transcript dataset",
    )
    parser.add_argument("--audio-dir", required=True, help="directory of .wav clips")
    parser.add_argument(
        "--transcript-dir",
        default=None,
        help="directory of <stem>.txt transcripts (default: same as --audio-dir)",
    )
    parser.add_argument("--language", default="en", help="ISO language code for all clips")
    parser.add_argument("--out", default="data/dataset", help="output directory for manifest.json")
    args = parser.parse_args(argv)

    try:
        entries = assemble_dataset(args.audio_dir, args.transcript_dir, language=args.language)
        manifest_path = write_manifest(entries, args.out)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    total = sum(e.duration_seconds for e in entries)
    print(f"assembled {len(entries)} entries ({total:.1f}s audio) -> {manifest_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
