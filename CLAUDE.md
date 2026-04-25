# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantMind is an intelligent knowledge extraction and retrieval framework for quantitative
finance. It is being repositioned as a domain library that runs on top of OpenAI Agents
SDK rather than as a self-contained agent framework. The next-step architecture
introduces these top-level modules:

- `flows/` — e2e processing pipelines (Agent runtime delegated to OpenAI Agents SDK)
- `knowledge/` — Pydantic-based knowledge schema standard
- `preprocess/` — fetching and formatting helpers (PDF/HTML → markdown, etc.)
- `mind/` — QuantMind's distinctive cognitive layer (working memory MVP first)
- `configs/` — centralized flow/input config types
- `magic.py` — natural-language → (input, cfg) resolver

Until those modules land, the repository is in a transitional state. PR1 removes the
self-built agent runtime so subsequent PRs can build the new architecture from a clean
slate.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -e .
```

### Code Quality
```bash
# Lint and format code
./scripts/lint.sh
# Or manually:
ruff format .
ruff check .

# Run tests
./scripts/unittest.sh
# Or manually:
pytest tests                         # all tests
pytest tests/quantmind/             # new quantmind tests
pytest tests/quantmind/models/      # specific module
```

## Current Modules (transitional, PR1)

After PR1's removal, the surviving modules are:

- **flow/**: Existing flow scaffolding (will be replaced by `flows/` in a later PR)
- **parsers/**: PDF / Llama parser helpers (will move to `preprocess/format/`)
- **sources/**: ArXiv source (fetch logic will move to `preprocess/fetch/`)
- **config/**: Configuration management (will be replaced by `configs/`)
- **llm/**: LLM block + embedding helpers (will be removed once `flow/` migrates)
- **models/**: `Paper`, `BaseContent`, `KnowledgeItem`, `analysis` (will move to `knowledge/`)
- **utils/**: `logger.py` (kept long-term) plus tmp helpers

These modules continue to compile and ship as-is in PR1; their replacements arrive in
PR2 (`knowledge/` + `configs/`), PR3 (`preprocess/`), and PR4 (`flows/` + drop `flow/` `llm/`).

## Key Dependencies

### Core Dependencies
- Pydantic for data validation
- PyMuPDF / Marker for PDF processing
- ArXiv API client
- LiteLLM for multi-provider LLM access
- YAML / Requests / httpx for configuration and IO

### Optional Dependencies
- OpenAI API
- llama-cloud-services (Llama parser)

## Development Guidelines

### Code Style
- Use Pydantic models for data validation
- Follow dependency injection patterns
- Use abstract base classes for extensibility
- Implement comprehensive error handling
- Write descriptive docstrings (Google style)

### Testing
- Unit tests in `tests/quantmind/`
- Mock external dependencies
- Test both success and failure cases
- Use pytest fixtures for common setups

### Configuration
- Use structured configuration via `quantmind.config.settings`
- Support environment variable overrides
- Validate configuration at startup
- Provide sensible defaults

### Architecture Principles
- **Separation of Concerns**: Each component has a single responsibility
- **Dependency Injection**: Components are configurable and testable
- **Pipeline Orchestration**: Workflow management with task dependencies
- **Quality Control**: Built-in deduplication and validation
- **Extensibility**: Easy to add new sources, parsers, taggers, storage

## User Development Guidance

- Config should add in `quantmind/config` (until `configs/` lands in a later PR)
- Data models should add in `quantmind/models` (until `knowledge/` lands)
- Do not use `Dict[str, Any]` in initialize functions — not type safe.
- Do not overdesign — implement the basic and straightforward code, refactor later.
- Tests in `tests/<module_name>/`, inherit `unittest.TestCase`.

## PR1 Cleanup (2026-04-25)

PR1 removes the self-built agent runtime to make room for the OpenAI Agents SDK
migration. Removed in PR1:

- `quantmind/brain/`, `quantmind/tools/`, `quantmind/storage/`, `quantmind/tagger/`
- `quantmind/models/{agent,memory,messages}.py`
- `quantmind/utils/{agentic_ext,monitoring}.py`
- vendored `smolagents/` and `LICENSE-APACHE`
- All examples (will be re-added per-flow in later PRs)

Preserved as historical reference: `archive/agent-runtime-final` branch.
