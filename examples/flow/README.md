# Flow Examples

This directory contains examples demonstrating the new QuantMind flow architecture.

## Examples Overview

### [01_custom_flow](./01_custom_flow/) - Simple Custom Flow

Learn how to create a custom flow from scratch:

- Basic flow configuration with Pydantic models
- YAML-based prompt templates
- Python-based orchestration logic
- Direct LLM block access

**Key Learning**: How to implement custom business logic using the new architecture.

### [02_summary_flow](./02_summary_flow/) - Built-in Summary Flow Demo

Explore the built-in SummaryFlow with various configurations:

- Chunking strategies (size-based, custom, disabled)
- Mixture mode (cheap + powerful LLMs)
- Mock LLM implementation for testing
- Real financial research paper examples

**Key Learning**: How to leverage and configure built-in flows for optimal cost/quality trade-offs.

### [03_podcast_flow](./03_podcast_flow/) - Built-in Podcast Flow Demo

Explore the built-in PodcastFlow with various configurations:

- Mixture mode (intro, main, outro)
- Mock LLM implementation for testing
- Real podcast script examples in various domains

**Key Learning**: How to leverage and configure built-in flows for optimal script in specific domain.

## Architecture Principles Demonstrated

Both examples showcase the core principles of the new flow architecture:

1. **Resource-Based Configuration**: Config defines resources (LLM blocks, templates), not orchestration logic
2. **Code-Based Orchestration**: Business logic implemented in Python for maximum flexibility
3. **Direct Access**: No unnecessary wrapper methods, just direct access to resources
4. **Template Separation**: Prompts in YAML files for easy editing and maintenance
5. **Type Safety**: Pydantic-based configuration instead of complex schemas

## Quick Start

Choose the example that matches your needs:

- **Want to create a custom flow?** → Start with `01_custom_flow`
- **Want to use built-in flows?** → Start with `02_summary_flow`

Each example is self-contained and includes:

- Complete working code
- Mock implementations (no API keys required)
- Comprehensive documentation
- Clear explanations of benefits

## Running Examples

Each example can be run independently:

```bash
# Custom flow example
cd 01_custom_flow
# Prepare the api-key .env file
cp .env.example .env
python pipeline.py
```

```bash
# Summary flow example
cd 02_summary_flow
# Prepare the api-key .env file
cp .env.example .env
python pipeline.py
```

```bash
# Podcast flow example
cd 03_podcast_flow
# Prepare the api-key .env file
cp .env.example .env
python pipeline.py
```

Both examples work without API keys by using mock LLM implementations that demonstrate the flow logic and architecture benefits.
