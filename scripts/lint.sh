#! /bin/bash

echo "Formatting code..."
ruff format .

echo "Checking code with linter..."
ruff check .
