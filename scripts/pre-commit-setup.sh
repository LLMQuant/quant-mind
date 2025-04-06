#!/bin/bash

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing pre-commit..."
pip install pre-commit

echo "Installing Git hooks..."
pre-commit install
pre-commit install --hook-type pre-push

echo "Pre-commit hooks successfully installed!"
