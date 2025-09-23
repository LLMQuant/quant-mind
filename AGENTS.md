# AI Development Guide

> The following context is a bundle of best practices for AI development. Please follow the guidance strictly.

## Avoid Meaningless Wrapper Methods

**Core Rule:** If a method only does simple data access + basic error checking, use it inline directly.

One-sentence Decision Criteria: **"Does this method do any actual work beyond wrapping the call?"**

### Simplified Comparison

```python
# ❌ Meaningless wrapper
def _get_llm(self, identifier: str):
    if identifier not in self._llm_blocks:
        raise KeyError(f"LLM block '{identifier}' not found")
    return self._llm_blocks[identifier]

# ✅ Direct usage
llm_block = self._llm_blocks[identifier]  # KeyError naturally thrown
```

### Keep wrappers when they provide

- **Complex logic**: Conditional logic, data transformation, loops
- **Abstract interfaces**: Abstract methods, public APIs
- **Side effects**: Logging, state changes, external calls

**Remember:** Encapsulation should hide complexity, not add complexity.

## Impl with Unit Test and Example

If you are implementing a new feature, please implement the unit test and example.

- For unit test, add in `tests/<module_name>`, and inherit the `unittest.TestCase` class.
- For example, add in `examples/<module_name>`, and just demo the simple usage. (do not add too many use cases in single file)

## Comment Style

- All comments should be in English.
- All comments should be in the Google style.
