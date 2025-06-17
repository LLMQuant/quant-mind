# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantMind is an intelligent knowledge extraction and retrieval framework for quantitative finance. It transforms unstructured financial content into a queryable knowledge graph using a two-stage architecture:

**Stage 1: Knowledge Extraction** (Current Implementation)
- Source APIs → Intelligent Parser → Workflow/Agent → Structured Knowledge Base
- Components: Crawlers, Parsers, Taggers, Workflow orchestration, Storage

**Stage 2: Intelligent Retrieval** (Future)
- Knowledge Base → Embeddings → Solution Scenarios (DeepResearch, RAG, Data MCP)

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

### QuantMind CLI Usage
```bash
# Basic extraction
quantmind extract "machine learning finance" --max-papers 10

# Full pipeline with storage
quantmind pipeline ml_pipeline "cat:q-fin.ST" --storage json --tagger rule

# Search stored papers
quantmind search --categories "Machine Learning in Finance" --limit 5

# System status
quantmind status

# Configuration management
quantmind config create --output config.yaml
quantmind config show
```

### Legacy System
```bash
# Old autoscholar system (still available)
python quant_scholar.py
```

## New Architecture (QuantMind v0.2.0)

### Core Modules

**quantmind/** - New modular architecture following Stage 1 design:

- **sources/**: Content acquisition layer
  - `base.py`: Abstract source interface
  - `arxiv_source.py`: ArXiv API integration with financial focus

- **parsers/**: Content processing layer
  - `base.py`: Abstract parser interface
  - `pdf_parser.py`: PDF extraction (PyMuPDF + Marker support)

- **tagger/**: Classification and labeling layer
  - `base.py`: Abstract tagger interface
  - `rule_tagger.py`: Rule-based financial classification
  - `llm_tagger.py`: LLM-powered advanced tagging

- **workflow/**: Orchestration layer
  - `agent.py`: Main WorkflowAgent for pipeline coordination
  - `pipeline.py`: Pipeline execution with dependency management
  - `tasks.py`: Task definitions (Crawl, Parse, Tag, Store)

- **storage/**: Knowledge base layer
  - `base.py`: Abstract storage interface
  - `json_storage.py`: JSON file-based storage with indexing

- **models/**: Data models
  - `paper.py`: Enhanced Paper model with Pydantic validation
  - `knowledge_graph.py`: Advanced graph operations

- **config/**: Configuration management
  - `settings.py`: Structured configuration with validation

- **utils/**: Shared utilities
  - `logger.py`: Consistent logging setup

### Examples and Usage

- **examples/quantmind/**: Complete usage examples
  - `basic_usage.py`: Basic pipeline demonstration
  - `config_example.py`: Configuration system demo

### Legacy System (autoscholar/)

Still available for backward compatibility:
- **crawler/**: Legacy crawlers
- **parser/**: Legacy parsers
- **knowledge/**: Legacy graph models
- **visualization/**: Pyvis visualizations

## Key Dependencies

### Core Dependencies
- Pydantic for data validation
- NetworkX for graph operations
- PyMuPDF for PDF processing
- ArXiv API client
- YAML for configuration
- Requests for HTTP operations

### Optional Dependencies
- OpenAI API (for LLM tagger)
- CAMEL-AI (alternative LLM framework)
- Marker (AI-powered PDF parsing)
- PyVis (graph visualization)

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

## Migration Notes

When migrating from autoscholar to quantmind:
1. Use `WorkflowAgent` instead of direct crawler usage
2. Configure components via `Settings` system
3. Use the CLI for common operations
4. Take advantage of new pipeline orchestration
5. Leverage improved error handling and logging
