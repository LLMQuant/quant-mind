# Use QuantMind

## Quick Summary

- **Purpose**: Route library users and agents to current public operations,
  inputs, results, examples, and guides.
- **Read when**: Calling QuantMind as a library or selecting a supported public
  operation.
- **Load next**: Start with the component row that matches the requested
  operation; do not load unrelated component designs.
- **Import rule**: Inputs/configs come from `quantmind.configs`, operations from
  `quantmind.flows`, and result types from their owning layer.

## Contents

- [Public Usage Sources](#public-usage-sources)
- [Import Boundary](#import-boundary)

## Public Usage Sources

Use this index when calling QuantMind as a library. These links point to the
current public API, focused examples, and component-specific guidance.

| Need | Canonical source |
|---|---|
| Current operations, inputs, results, and sources | [Public component catalog](../../docs/README.md) |
| Installation and common usage | [Root README usage](../../README.md#-usage-examples) |
| Paper extraction | [Paper E2E design contract](../design/flow/paper.md) |
| News collection | [News design and behavior](../design/flow/news.md) |
| Local semantic search | [Library guide](../../docs/library.md) and [focused example](../../examples/library/README.md) |
| Runnable operation examples | [`examples/flows/`](../../examples/flows/) |
| Focused preprocessing examples | [`examples/preprocess/`](../../examples/preprocess/) |

## Import Boundary

Import public inputs and configs from `quantmind.configs`, public operations
from `quantmind.flows`, and result contracts from the owning layer identified
by the component catalog.
