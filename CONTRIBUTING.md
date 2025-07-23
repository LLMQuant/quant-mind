# Contributing to QuantMind

Thank you for your interest in contributing to QuantMind! This guide outlines the best practices for contributing to our intelligent knowledge extraction framework for quantitative finance.

## Quick Start

1. **Fork and clone** the repository
2. **Set up development environment**:

   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

3. **Install pre-commit hooks**:

   ```bash
   # Automated setup (recommended)
   ./scripts/pre-commit-setup.sh

   # Or manual setup
   pip install pre-commit
   pre-commit install
   pre-commit install --hook-type pre-push
   ```

## Development Standards

### Code Quality

- **Location**: All new code must be in the `quantmind/` module
- **Style**: Google-style docstrings, 80-character line length
- **Architecture**: Use abstract base classes and dependency injection
- **Type Safety**: Comprehensive type hints with Pydantic models

### Testing Requirements

- **Unit tests**: Required for all new functionality in `tests/quantmind/`
- **Structure**: Mirror the `quantmind/` module structure
- **Coverage**: Test both success and error cases
- **Dependencies**: Mock external APIs and file systems

### Documentation

- **Examples**: Add practical usage examples to `examples/quantmind/`
- **Comments**: Clear inline documentation for complex logic
- **Docstrings**: Google-style format for all public methods

## Automated Quality Checks

Our pre-commit configuration (`.pre-commit-config.yaml`) enforces quality standards:

### On Every Commit

- **Formatting**: `ruff format` (80-char line length)
- **Linting**: `ruff check --fix` (auto-fix issues)
- **File Quality**: Trailing whitespace, EOF, YAML syntax
- **Safety**: Check for large files and merge conflicts

### On Push to Remote

- **Unit Tests**: Full test suite via `scripts/unittest.sh`

### Manual Execution

```bash
# Run formatting and linting
./scripts/lint.sh

# Run specific tests
./scripts/unittest.sh tests/quantmind/sources/
./scripts/unittest.sh all  # Run all tests
```

## Contribution Types

### 1. New Sources

- Extend `BaseSource[ContentType]` in `quantmind/sources/`
- Add corresponding config in `quantmind/config/sources.py`
- Include comprehensive tests in `tests/sources/`
- Provide usage example in `examples/sources/`

### 2. New Parsers

- Extend `BaseParser` in `quantmind/parsers/`
- Handle multiple content formats
- Include error handling and validation
- Add tests for different input types

### 3. New Taggers

- Extend `BaseTagger` in `quantmind/tagger/`
- Support both rule-based and ML-based approaches
- Include configuration options
- Test with various content types

### 4. Storage Backends

- Extend `BaseStorage` in `quantmind/storage/`
- Implement indexing and querying
- Handle concurrent access
- Include performance tests

## Code Quality Guidelines

### Architecture Principles

- **Separation of Concerns**: Single responsibility per component
- **Dependency Injection**: Configurable and testable components
- **Error Handling**: Comprehensive logging and graceful failures
- **Type Safety**: Use Pydantic models and type hints

### Best Practices

- Use existing utilities (`quantmind.utils.logger`)
- Follow the workflow orchestration pattern
- Implement proper configuration validation
- Add deduplication and quality control

## Pull Request Process

1. **Create feature branch** from `master`
2. **Implement changes** following development standards
3. **Pre-commit hooks will automatically**:
   - Format code with `ruff format`
   - Fix linting issues with `ruff check --fix`
   - Validate file quality and syntax
4. **Before pushing**:

   ```bash
   # Verify all checks pass locally
   pre-commit run --all-files
   ./scripts/unittest.sh all
   ```

5. **Submit PR** with clear description and test plan

### PR Requirements

- [ ] Code in `quantmind/` module following architecture patterns
- [ ] Unit tests in `tests/` with comprehensive coverage
- [ ] Usage example in `examples/` (for new features)
- [ ] All pre-commit hooks pass (automatic on commit/push)
- [ ] Clear commit messages and PR description

## Development Tips

### Testing

```bash
# Run specific test modules
pytest tests/quantmind/sources/
pytest tests/quantmind/models/

# Run with coverage
pytest tests --cov=quantmind
```

### Local Development

```bash
# Test CLI functionality
quantmind extract "test query" --max-papers 5
quantmind config show
```

### Configuration

- Use structured config via `quantmind.config.settings`
- Support environment variable overrides
- Provide sensible defaults
- Validate at startup

## Questions?

- Check existing issues and discussions
- Follow the patterns in existing code
- Look at `examples/` for usage patterns
- Review the architecture in `CLAUDE.md`

Thank you for contributing to QuantMind! 🚀
