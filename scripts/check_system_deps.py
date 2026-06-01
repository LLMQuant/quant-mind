#!/usr/bin/env python3
"""Check system-level (non-Python) dependencies for QuantMind.

Python deps are managed by ``pyproject.toml`` + ``uv pip install``.
This script enumerates the *additional* system tools certain optional
features rely on at runtime, prints their status, and exits non-zero
when a **required** dep is missing. Optional deps are reported with
their install hint but do not fail the check.

**Adding a new dependency:** append one entry to ``_DEPS``. The
README install flow does not need to change.

Standalone:

    python scripts/check_system_deps.py

From the setup script:

    bash scripts/setup.sh   # also calls this at the end
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemDep:
    """One non-Python external tool QuantMind may need at runtime."""

    name: str  # human display name
    binary: str  # executable to look for on PATH
    version_args: tuple[str, ...]  # arguments that print a version
    feature: str  # which QuantMind feature requires it
    required: bool  # True = blocks setup; False = informational
    install_hint: str  # short install hint shown when missing


_DEPS: tuple[SystemDep, ...] = (
    SystemDep(
        name="Node.js",
        binary="node",
        version_args=("--version",),
        feature="FilesystemMemory (MCP filesystem server)",
        required=False,
        install_hint=(
            "brew install node  # macOS\n"
            "    sudo apt-get install -y nodejs  # Debian/Ubuntu"
        ),
    ),
    SystemDep(
        name="npx",
        binary="npx",
        version_args=("--version",),
        feature="FilesystemMemory (MCP filesystem server)",
        required=False,
        install_hint="installed alongside Node.js",
    ),
    # PR7+ will likely append: SystemDep("sqlite-vec", ...), etc.
)


def _probe(dep: SystemDep) -> tuple[bool, str]:
    """Return ``(found, info)`` for a single dependency."""
    path = shutil.which(dep.binary)
    if path is None:
        return False, "not found on PATH"
    try:
        completed = subprocess.run(
            [path, *dep.version_args],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"probe failed: {exc}"
    if completed.returncode != 0:
        return False, f"version check failed (rc={completed.returncode})"
    output = completed.stdout or completed.stderr
    first_line = (
        output.strip().splitlines()[0]
        if output.strip()
        else "(no version output)"
    )
    return True, first_line


def main() -> int:
    """Run the audit and return a process exit code (0 = OK, 1 = blocker)."""
    print("Checking QuantMind system dependencies\n")
    missing_required: list[SystemDep] = []
    missing_optional: list[SystemDep] = []

    for dep in _DEPS:
        ok, info = _probe(dep)
        if ok:
            status = "✓ OK"
        elif dep.required:
            status = "✗ MISSING (required)"
            missing_required.append(dep)
        else:
            status = "· MISSING (optional)"
            missing_optional.append(dep)
        print(f"  {status:<22s} {dep.name:<10s}  {info}")
        print(f"  {'':<22s}   used by: {dep.feature}")
        if not ok:
            for line in dep.install_hint.splitlines():
                print(f"  {'':<22s}   install: {line}")
        print()

    if missing_required:
        names = ", ".join(d.name for d in missing_required)
        print(f"✗ {len(missing_required)} required dep(s) missing: {names}")
        return 1

    if missing_optional:
        names = ", ".join(d.name for d in missing_optional)
        print(
            f"All required deps OK. Optional missing: {names} "
            "(install only if you use the listed features)."
        )
    else:
        print("All system deps present (required + optional).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
