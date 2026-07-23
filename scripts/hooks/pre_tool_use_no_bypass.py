#!/usr/bin/env python3
"""PreToolUse guard: stop an agent from bypassing the repository's git hooks.

Shared by Claude Code (``.claude/settings.json``) and Codex
(``.codex/hooks.json``); both invoke this same script and speak the same
stdin-JSON / stdout-JSON hook protocol. The guard reads the tool event on
stdin and, for a Bash command that actually runs ``git ... --no-verify``
(which would skip the ``commit-msg`` / ``pre-commit`` / ``pre-push`` checks),
returns a ``deny`` decision. Everything else passes through untouched.

Matching is precise, not a substring scan. The command is tokenized with
``shlex`` and split into simple commands on shell operators, so ``--no-verify``
only triggers a deny when it is a real argument token of a real ``git``
invocation. A command that merely *mentions* the flag inside a quoted string
— an ``echo``, a ``grep`` pattern, a commit message, a JSON payload — keeps it
inside a single token and passes. Residual gap: shell indirection such as
``$VAR`` expansion or ``eval`` is not resolved (a deliberately obfuscated
bypass is out of scope for a mechanical guard).

Design: mechanical check only, never semantic judgement. It fails open — any
parse error yields no decision — so a malformed event can never wedge the
agent. This is the one guarantee git hooks cannot self-enforce (``--no-verify``
is, by definition, the flag that turns them off), so it lives at the agent
layer instead.
"""

import json
import shlex
import sys

# Words that may precede the real executable in a simple command.
_PREFIX_WORDS = frozenset(
    {"command", "env", "sudo", "nice", "nohup", "time", "builtin", "exec"}
)
# Bare operator tokens shlex(punctuation_chars=True) emits between commands.
_OPERATOR_CHARS = frozenset("();<>|&")


def _is_operator(token: str) -> bool:
    """Return True if ``token`` is a run of shell operator characters."""
    return bool(token) and all(ch in _OPERATOR_CHARS for ch in token)


def _is_assignment(token: str) -> bool:
    """Return True if ``token`` is a leading ``VAR=value`` assignment."""
    eq = token.find("=")
    return eq > 0 and not token.startswith("-") and token[:eq].isidentifier()


def _git_argvs(command: str) -> list[list[str]]:
    """Return the argv of every ``git`` simple-command inside ``command``.

    Raises ``ValueError`` on unbalanced quotes so the caller can fail open.
    """
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    tokens = list(lexer)

    simple: list[str] = []
    simples: list[list[str]] = [simple]
    for token in tokens:
        if _is_operator(token):
            simple = []
            simples.append(simple)
        else:
            simple.append(token)

    git_argvs: list[list[str]] = []
    for cmd in simples:
        start = 0
        while start < len(cmd) and (
            _is_assignment(cmd[start]) or cmd[start] in _PREFIX_WORDS
        ):
            start += 1
        if start < len(cmd) and cmd[start] == "git":
            git_argvs.append(cmd[start:])
    return git_argvs


_REASON = (
    "Blocked: this command runs git with --no-verify, which skips the "
    "repository's commit-msg / pre-commit / pre-push checks. Do not bypass "
    "verification. Fix the underlying issue (formatting, lint, types, tests, "
    "commit message) and run the command again without that flag. If bypassing "
    "is genuinely required, ask the user to authorize it explicitly for this "
    "one command."
)


def evaluate(payload: dict) -> dict | None:
    """Return a deny decision for a real ``git --no-verify`` command, else None."""
    if payload.get("tool_name") != "Bash":
        return None
    command = (payload.get("tool_input") or {}).get("command", "")
    if not isinstance(command, str) or not command:
        return None
    try:
        git_argvs = _git_argvs(command)
    except ValueError:
        return None  # unbalanced quotes etc — fail open
    if any("--no-verify" in argv for argv in git_argvs):
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
