#!/usr/bin/env bash
# QuantMind Golden Harness — single command, fast-fail verify loop.
#
# Run before pushing; CI runs the same script (.github/workflows/verify.yml).
# Steps execute in order; the first failure stops the loop.
#
# Steps:
#   1. ruff format --check    formatting must be clean
#   2. ruff check             lint must pass
#   3. basedpyright           type check must pass
#   4. lint-imports           architectural boundary contracts must hold
#   5. pytest --cov           tests + coverage floor (configured in pyproject)

set -euo pipefail

echo "==> [1/5] ruff format --check"
ruff format --check .

echo "==> [2/5] ruff check"
ruff check .

echo "==> [3/5] basedpyright"
basedpyright

echo "==> [4/5] lint-imports"
lint-imports

echo "==> [5/5] pytest --cov"
pytest

echo
echo "[OK] verify loop passed"
