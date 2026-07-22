# Writing tests for QuantMind

The canonical test standard. Read this before adding or changing tests. It
covers where tests live, why the default suite is offline, coverage, how live
behavior is verified, and the change→test obligations that keep coverage honest.

## Where tests live

- Mirror the module under test: `tests/<module>/test_<topic>.py`.
- Subclass `unittest.TestCase`, or `unittest.IsolatedAsyncioTestCase` for
  `async` code. Everything runs via `pytest`.
- Put shared builders and fixtures in a module helper (e.g.
  `tests/paper_helpers.py`), not inline in every test.

## The default suite is offline and deterministic

- No real network, no real LLM, no dependence on wall-clock or randomness.
- Mock only across a boundary: patch the SDK runner (`Runner.run` or
  `run_with_observability`) for agents, patch the HTTP client for fetchers, use
  an in-memory (`:memory:`) store. Use real objects for in-package types.
- Cover the success path **and** at least one failure path per public function
  (typed errors, boundary rejection, timeout).
- Assert observable behavior and values, not implementation details you expect
  to refactor.

## Coverage

- `pytest --cov=quantmind --cov-fail-under=75` runs inside `scripts/verify.sh`.
  New code must not lower branch coverage.
- Coverage measures `quantmind/` only — `scripts/` and `tests/` are not
  measured. Adding a test that only exists to raise a `scripts/` file's coverage
  is pointless (see the next section).

## Live / end-to-end behavior

- Real-network or real-LLM behavior belongs in a bounded script
  `scripts/verify_<component>_e2e.py`: gated on the required credential
  (skip-not-fail when it is unset), bounded by `asyncio.wait_for`, over a small
  fixture, ending in a clear PASS/FAIL line. Wire it into
  `.github/workflows/e2e.yml` (path filters + schedule).
- **Do not unit-test a script.** No offline test should import a
  `scripts/verify_*` module to assert its constants, model list, or wiring —
  that only restates the source, rots on every edit, and adds no real coverage.
  An e2e script is validated by *running* it in the e2e workflow, not by a
  meta-test.

## Best practices

- One behavior per test; name it after the guarantee, not the mechanism.
- Reproduce a bug with a failing test **before** fixing it, then keep it as the
  regression.
- For cross-provider or structured-output code, mock the SDK runner and assert
  every branch offline — the strict path, the fallback path, and that unrelated
  errors propagate — rather than reaching a real provider.
- Prefer table-free explicitness: a reader should see the input and the expected
  outcome in the test body.

## Change → test obligations

| When you ... | Add or update |
|---|---|
| Add or change a public callable / flow | Offline success + failure tests in `tests/<module>/`; a magic-introspection test if it follows `(input, *, cfg)`. |
| Add a knowledge schema | Validation success + failure, plus a dump/load round-trip. |
| Add cross-provider or LLM-call behavior | Offline tests with the SDK runner mocked, covering every branch (e.g. strict output + json-object fallback + error propagation). |
| Add a public-network source or parser | Offline mocked parse / boundary / continuation tests **and** a `scripts/verify_<component>_e2e.py` slice. |
| Fix a bug | A regression test that fails before the fix and passes after. |
| Add or edit a `contexts/**/*.md` page | Keep `## Quick Summary` before `## Contents`, the Contents anchors matching the `##` sections, and any index links resolving — `tests/test_contexts.py` enforces this. |

## Required check

`bash scripts/verify.sh` (ruff, ruff-format, basedpyright, import-linter,
`pytest --cov`) must pass before every push. CI runs the same harness.
