"""One-off local check: confirm the 4 source repos are actually archived.

Not run in CI (the sibling `portfolio-archived-repos` checkout and the
portfolio-level `PORTFOLIO_INDEX.md` only exist in a full local checkout of
the portfolio, not in this repo's own CI, which checks out only
`voice-ai-suite`). The reusable detection logic this exercises is
unit-tested against fixtures in `tests/test_archival.py`.

Usage:
    python scripts/verify_archival.py [portfolio-root]

`portfolio-root` defaults to the parent of this repo.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voice_core.archival import check_archival  # noqa: E402

SUITE_NAME = "voice-ai-suite"
ORIGINALS = [
    "realtime-voice-agent",
    "voice-agent-evaluation",
    "speech-model-evaluator",
    "python-voice-rag-chatbot",
]


def main() -> int:
    portfolio_root = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    )
    archived_root = portfolio_root.parent / "portfolio-archived-repos"
    index_path = portfolio_root / "PORTFOLIO_INDEX.md"

    if not index_path.exists():
        print(f"SKIP: {index_path} not found (not a full portfolio checkout)")
        return 0

    index_text = index_path.read_text(encoding="utf-8")
    all_ok = True
    for original in ORIGINALS:
        readme_path = archived_root / original / "README.md"
        readme_text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
        status = check_archival(
            suite_name=SUITE_NAME,
            original_name=original,
            original_readme_text=readme_text,
            index_text=index_text,
        )
        marker = "OK" if status.complete else "FAIL"
        print(f"{marker}: {original} (readme_archived={status.readme_archived}, index_ok={status.index_ok})")
        all_ok = all_ok and status.complete

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
