# Contributing to QuantMind

Thank you for contributing to QuantMind! This guide provides essential information for developers.

## 🚀 Quick Setup

1. **Fork and clone** the repository
2. **Set up environment**:
   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e .
   ```
3. **Install pre-commit hooks**:
   ```bash
   ./scripts/pre-commit-setup.sh
   ```

## 🛠️ Development Setup

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality and consistency. These hooks automatically format code, run linting, and perform other quality checks before each commit.

**Install pre-commit hooks:**

```bash
# Automated setup (recommended)
./scripts/pre-commit-setup.sh

# Or manual setup
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push
```

**What the hooks do:**

- **On every commit:**
  - Code formatting with `ruff format` (80-char line length)
  - Linting with `ruff check --fix` (auto-fixes issues)
  - File quality checks (trailing whitespace, EOF, YAML syntax)
  - Safety checks (large files, merge conflicts)

- **On push to remote:**
  - Full unit test suite via `scripts/unittest.sh`

**Manual execution:**

```bash
# Run formatting and linting
./scripts/lint.sh

# Run all pre-commit hooks on all files
pre-commit run --all-files

# Run specific tests
./scripts/unittest.sh tests/quantmind/sources/
./scripts/unittest.sh all  # Run all tests
```

**Troubleshooting:**

- If hooks fail, fix the issues and commit again
- To skip hooks temporarily (not recommended): `git commit --no-verify`
- Update hooks: `pre-commit autoupdate`

## 📝 Development Standards

### Code Requirements
- **Location**: All new code in `quantmind/` module
- **Style**: Google-style docstrings, 80-char line length
- **Architecture**: Abstract base classes + dependency injection
- **Type Safety**: Pydantic models + comprehensive type hints

### Testing
- **Unit tests**: Required in `tests/quantmind/` (mirror module structure)
- **Coverage**: Test success and error cases
- **Mocking**: Mock external APIs and file systems

### Documentation
- **Examples**: Add to `examples/quantmind/` for new features
- **Docstrings**: Google-style format for all public methods

## 🏗️ Contribution Types

### New Sources
- Extend `BaseSource[ContentType]` in `quantmind/sources/`
- Add config in `quantmind/config/sources.py`
- Include tests and usage example

### New Parsers
- Extend `BaseParser` in `quantmind/parsers/`
- Handle multiple content formats with error handling

### New Taggers
- Extend `BaseTagger` in `quantmind/tagger/`
- Support rule-based and ML approaches

### Storage Backends
- Extend `BaseStorage` in `quantmind/storage/`
- Implement indexing, querying, and concurrent access

## 🔄 Pull Request Process

1. **Create feature branch** from `master`
2. **Follow conventional commits**: `type(scope): description`
3. **Pre-commit hooks** run automatically on commit/push
4. **Before submitting**:
   ```bash
   pre-commit run --all-files
   ./scripts/unittest.sh all
   ```
5. **Submit PR** using our template

### PR Checklist
- [ ] Code in `quantmind/` following architecture patterns
- [ ] Unit tests with comprehensive coverage
- [ ] Usage example (for new features)
- [ ] All pre-commit hooks pass
- [ ] Conventional commit format

## 💡 Development Tips

```bash
# Run specific tests
pytest tests/quantmind/sources/
pytest tests/quantmind/models/

# Test CLI functionality
quantmind extract "test query" --max-papers 5
quantmind config show

# Check code quality
./scripts/lint.sh
```

## ❓ Questions?

- Check existing [issues](https://github.com/LLMQuant/quant-mind/issues)
- Review architecture patterns in existing code
- Look at `examples/` for usage patterns
- See `CLAUDE.md` for detailed architecture

Thank you for contributing! 🚀
