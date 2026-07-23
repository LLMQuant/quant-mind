#!/usr/bin/env python3
"""PreToolUse guard: stop an agent from bypassing the repository's git hooks.

Shared by Claude Code (``.claude/settings.json``) and Codex
(``.codex/hooks.json``); both invoke this same script and speak the same
stdin-JSON / stdout-JSON hook protocol. The guard reads the tool event on
stdin and, for a Bash ``git`` command that carries ``--no-verify`` (which
would skip the ``commit-msg`` / ``pre-commit`` / ``pre-push`` checks), returns
a ``deny`` decision. Everything else passes through untouched.

Design: mechanical check only, never semantic judgement. It fails open — any
parse error yields no decision — so a malformed event can never wedge the
agent. This is the one guarantee git hooks cannot self-enforce (``--no-verify``
is, by definition, the flag that turns them off), so it lives at the agent
layer instead.
"""

import json
import re
import sys

_GIT = re.compile(r"\bgit\b")
_NO_VERIFY = re.compile(r"(?:^|\s)--no-verify(?:\s|$)")

_REASON = (
    "Blocked: this command uses --no-verify, which skips the repository's "
    "commit-msg / pre-commit / pre-push checks. Do not bypass verification. "
    "Fix the underlying issue (formatting, lint, types, tests, commit "
    "message) and run the command again without --no-verify. If bypassing is "
    "genuinely required, ask the user to authorize it explicitly for this "
    "one command."
)


def evaluate(payload: dict) -> dict | None:
    """Return a deny decision for a --no-verify git command, else ``None``."""
    if payload.get("tool_name") != "Bash":
        return None
    command = (payload.get("tool_input") or {}).get("command", "")
    if not isinstance(command, str):
        return None
    if _GIT.search(command) and _NO_VERIFY.search(command):
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": _REASON,
            }
        }
    return None


def main() -> int:
    """Read the tool event from stdin; print a deny decision if warranted."""
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # fail open
    if not isinstance(payload, dict):
        return 0
    decision = evaluate(payload)
    if decision is not None:
        print(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
