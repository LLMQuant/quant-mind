#!/usr/bin/env bash
# QuantMind one-shot setup: Python deps via uv + system dep audit.
#
# Idempotent — safe to re-run after pulling new dependencies. Adding
# new external (non-Python) deps means appending one row to
# scripts/check_system_deps.py; this script does not need to change.
#
# Usage:
#   bash scripts/setup.sh

set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "[ERROR] uv not on PATH. Install it first:"
  echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "    # or:  pip install uv"
  exit 1
fi

echo "==> [1/3] Creating venv (.venv) if missing"
uv venv

# Bind uv to *this project's* venv explicitly. Without this, `uv pip
# install` may resolve to a different interpreter (e.g., an active
# conda env) and install there instead, which leaves .venv empty.
export VIRTUAL_ENV="$PWD/.venv"
PY="$VIRTUAL_ENV/bin/python"

echo
echo "==> [2/3] Installing Python deps (editable, with dev extras)"
uv pip install --python "$PY" -e ".[dev]"

echo
echo "==> [3/3] Checking system (non-Python) dependencies"
# We don't fail setup on *optional* deps; check_system_deps.py exits
# non-zero only when a required dep is missing.
"$PY" scripts/check_system_deps.py

echo
echo "==> Setup complete."
echo "    source .venv/bin/activate"
echo "    bash scripts/verify.sh   # full local check before push"
