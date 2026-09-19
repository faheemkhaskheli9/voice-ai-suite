"""Verification helpers for Phase 6 (archive the 4 source repos).

This module doesn't perform the archival itself (that's a one-time,
filesystem-level move of another repo's directory — not something to
automate inside *this* package). It gives the acceptance criteria a
checkable, testable shape:

- a source repo's README carries the archived-status badge
- the portfolio index points readers at this suite instead of the originals

The unit tests exercise this logic against small in-memory fixtures (so they
run in any checkout, including CI where the sibling `portfolio-archived-repos`
directory isn't present). Confirming it against the *real* archived repos and
the real `PORTFOLIO_INDEX.md` is a local, one-off check — see
`scripts/verify_archival.py`.

Pattern: reused clinical-llm-suite's/nlp-classification-suite's
archival-verification module (same checkable-acceptance-criteria shape,
adapted to 4 originals here instead of 3/2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

ARCHIVED_BADGE_MARKER = "status-archived"


def repo_readme_is_archived(readme_text: str) -> bool:
    """True if a repo's README carries the archived-status badge."""
    return ARCHIVED_BADGE_MARKER in readme_text


def index_references_suite(index_text: str, suite_name: str) -> bool:
    """True if the portfolio index mentions `suite_name` at all."""
    return suite_name in index_text


_CONTEXT_WINDOW_CHARS = 150
_ARCHIVAL_CONTEXT_WORDS = ("archived", "moved", "combined", "original")


def index_lists_original_as_active_project(index_text: str, original_name: str) -> bool:
    """True if `original_name` still appears as its own active-project entry.

    A mention inside an archival note ("...combined originals
    (`original_name`, ...) ... moved out ... to portfolio-archived-repos")
    is expected and fine, even when the note wraps across markdown lines.
    What would be wrong is `original_name` appearing with no
    "archived"/"moved"/"combined"/"original" wording anywhere nearby — the
    way an active repo is listed as its own bullet/heading.
    """
    for match in re.finditer(re.escape(original_name), index_text):
        start = max(0, match.start() - _CONTEXT_WINDOW_CHARS)
        end = min(len(index_text), match.end() + _CONTEXT_WINDOW_CHARS)
        window = index_text[start:end].lower()
        if not any(word in window for word in _ARCHIVAL_CONTEXT_WORDS):
            return True
    return False


@dataclass(frozen=True)
class ArchivalStatus:
    """Result of checking one original repo's archival state."""

    original_name: str
    readme_archived: bool
    index_ok: bool

    @property
    def complete(self) -> bool:
        return self.readme_archived and self.index_ok


def check_archival(
    *,
    suite_name: str,
    original_name: str,
    original_readme_text: str,
    index_text: str,
) -> ArchivalStatus:
    """Check one original repo's archival state against these 2 documents."""
    readme_archived = repo_readme_is_archived(original_readme_text)
    index_ok = index_references_suite(index_text, suite_name) and not (
        index_lists_original_as_active_project(index_text, original_name)
    )
    return ArchivalStatus(
        original_name=original_name,
        readme_archived=readme_archived,
        index_ok=index_ok,
    )
