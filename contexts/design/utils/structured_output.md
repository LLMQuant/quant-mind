# Cross-provider structured output

## Quick Summary

- **Purpose**: Get a validated Pydantic object back from an Agents SDK run on
  any provider the SDK can route, including ones that reject strict
  `json_schema` structured output.
- **Read when**: Adding or changing an agent call that sets `output_type=`, or
  debugging a `BadRequestError` about `response_format` on a non-OpenAI model.
- **Owner**: `quantmind/utils/structured_output.py` (`run_structured`). A leaf
  helper; every package may import it.
- **Status**: Current behavior. One bounded fallback — no retry loop, no
  provider table.

## Contents

- [Motivation](#motivation)
- [The Fallback Ladder](#the-fallback-ladder)
- [Layer Boundary](#layer-boundary)
- [Non-Goals](#non-goals)
- [Verification](#verification)

## Motivation

Setting `output_type=` makes the SDK send a strict
`response_format={"type": "json_schema"}`. Native OpenAI routes accept it, but
some LiteLLM-routed providers (DeepSeek, ...) reject that `response_format`
outright — and they reject it at *request* time, before any output exists, so a
parse-stage retry cannot recover. The request itself must change: drop
`output_type`, ask for `{"type": "json_object"}`, and validate the raw output in
code. This must stay invisible to callers, who pass only `cfg.model`.

## The Fallback Ladder

`run_structured(output_type, *, build_agent, run)` runs one strict-first ladder:

1. Run the agent from `build_agent(False)` — it carries `output_type=`, so the
   SDK sends strict `json_schema` and returns a parsed model.
2. If that raises a `BadRequestError` whose message names the `response_format`
   / `json_schema` (a narrow check), run the agent from `build_agent(True)` — no
   `output_type`, the JSON Schema pinned into the instructions, and
   `response_format` forced to `json_object` — then validate the raw string
   locally. `json_object_instructions` / `json_object_model_settings` build that
   agent; a fenced-JSON strip guards Markdown wrappers.

A `BadRequestError` unrelated to `response_format` is re-raised unchanged, never
masked by the fallback. Incapability is discovered by the provider's own
rejection, so a capable provider always keeps the stronger strict contract.

## Layer Boundary

The helper lives in `utils` (a leaf every package may import) and takes two
callbacks so it owns no runtime policy:

- `build_agent(json_object)` — how the agent is constructed at this call site
  (name, instructions, tools, model settings).
- `run(agent)` — which runner executes it. `flows` passes
  `run_with_observability`; `mind` passes its own `Runner.run` + `RunConfig`.

This is why `mind` never imports `flows`: the shared ladder is in `utils`, and
each layer injects its own runner. No new runtime module is introduced.

## Non-Goals

- No provider registry or model-name prefix table — detection is the provider's
  own `BadRequestError`, not a static capability list. (`litellm`'s own
  `supports_response_schema` is unreliable, e.g. it marks `openrouter/openai/*`
  routes as unsupported.)
- No unbounded retry — the ladder is exactly two rungs. A malformed
  `json_object` result surfaces its `ValidationError`; it is not re-prompted in
  a loop.
- No second, tool-less "salvage" agent and no fuzzy output repair (alias
  remapping, UUID regex). Strict schema on capable providers removes the need.

## Verification

Offline (`tests/utils/test_structured_output.py`, plus the two call sites'
tests): strict is the default; a simulated `response_format` rejection falls
back to `json_object` and validates a fenced or bare string; an unrelated
`BadRequestError` propagates unswallowed. Live: `scripts/verify_structure_e2e.py`
exercises a real `json_object`-only provider (DeepSeek) and a strict baseline
(GPT).
