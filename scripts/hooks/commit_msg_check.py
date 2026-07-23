#!/usr/bin/env python3
"""commit-msg hook: enforce an English, Conventional Commit subject line.

Invoked by pre-commit's ``commit-msg`` stage with the path to the commit
message file as ``sys.argv[1]``. The commit is blocked (non-zero exit) when
the subject line is not a Conventional Commit or contains CJK characters.

Stdlib-only and network-free, so it runs identically for a human, for Claude
Code, and for Codex (each spawns git, which invokes this hook). The accepted
type set mirrors ``.claude/skills/quantmind-dev/references/commit.md``; update
that canonical reference and this list together.
"""

import re
import sys
from pathlib import Path

# Types mirror commit.md. Keep the two in sync (commit.md is the source).
_TYPES = ("feat", "fix", "refactor", "docs", "test", "chore")
_CONVENTIONAL = re.compile(rf"^(?:{'|'.join(_TYPES)})(?:\([^)]+\))?!?: .+")

# Git-generated subjects that are not authored Conventional Commits.
_EXEMPT_PREFIXES = ("Merge ", "Revert ", "fixup! ", "squash! ", "amend! ")

# CJK ranges: symbols/punctuation, hiragana/katakana, CJK ideographs, and the
# full-width/half-width forms block. A subject with any of these is not English.
_CJK = re.compile(r"[　-〿぀-ヿ㐀-䶿一-鿿＀-￯]")


def subject_of(message: str) -> str:
    """Return the first non-empty, non-comment line of a commit message."""
    for line in message.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def check_subject(subject: str) -> list[str]:
    """Return a list of rule violations for ``subject`` (empty means valid)."""
    if not subject or subject.startswith(_EXEMPT_PREFIXES):
        return []
    errors: list[str] = []
    if _CJK.search(subject):
        errors.append(
            "contains non-English (CJK) characters; write it in English"
        )
    if not _CONVENTIONAL.match(subject):
        errors.append(
            "is not a Conventional Commit; use "
            "`<type>(<scope>): <summary>` with type in "
            f"{{{', '.join(_TYPES)}}}"
        )
    return errors


def main() -> int:
    """Validate the commit message file git passes; return an exit code."""
    if len(sys.argv) < 2:
        # No message file: nothing to check, do not block.
        return 0
    try:
        message = Path(sys.argv[1]).read_text(encoding="utf-8")
    except OSError:
        return 0
    subject = subject_of(message)
    errors = check_subject(subject)
    if not errors:
        return 0
    print("Commit message rejected. Subject line:", file=sys.stderr)
    print(f"  {subject!r}", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)
    print(
        "See .claude/skills/quantmind-dev/references/commit.md. "
        "Fix the message and commit again; do not bypass with --no-verify.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
