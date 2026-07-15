# Repository Label Guidance

This file is the canonical decision guide for labeling QuantMind issues and
pull requests. Labels describe three independent dimensions:

- `type:` answers what kind of change is proposed or delivered.
- `area:` answers which repository ownership surface is affected.
- `impact:` identifies special compatibility or live-network risk.

Choose labels from the work's intent and accepted scope, not only from the
files it happens to touch.

## Cardinality

Every issue and pull request must have **exactly one** `type:` label.

Normally choose **one** `area:` label. Use two only when the work genuinely
spans two ownership surfaces; never use more than two.

`impact:` labels are optional and additive. Resolution and community labels
do not count toward these limits.

## Type Labels

| Label | Use when |
|---|---|
| `type: bug` | Existing behavior does not meet its contract or documented expectation. |
| `type: feature` | The work adds a new capability or observable behavior. |
| `type: refactor` | Existing implementation is reorganized without intentionally changing public behavior. |
| `type: docs` | The primary deliverable is documentation, guidance, or explanatory example text. |
| `type: maintenance` | The work performs specific repository upkeep, such as dependency, configuration, release, or housekeeping changes. |
| `type: design` | The primary deliverable is an architecture decision, RFC, or design that precedes implementation. |

`type: maintenance` is not a miscellaneous bucket. If the intended outcome is
unclear, clarify the work before labeling it.

## Area Labels

| Label | Ownership surface |
|---|---|
| `area: contexts` | Repository information routing and discovery under `contexts/`. This does not mean all documentation. |
| `area: harness` | Contributor and agent controls: `AGENTS.md`, `CLAUDE.md`, skills, CI, verification scripts, hooks, templates, and rulesets. |
| `area: knowledge` | Knowledge models, schemas, serialization, and representation under `quantmind/knowledge/`. |
| `area: configs` | Typed inputs and configuration contracts under `quantmind/configs/`. |
| `area: preprocess` | Deterministic acquisition, parsing, cleaning, formatting, and source handling under `quantmind/preprocess/`. |
| `area: flows` | Public operation implementations under `quantmind/flows/`. It is not a generic synonym for pipeline or orchestration. |
| `area: mind` | Memory, tools, MCP integration, and the cognitive layer under `quantmind/mind/`. |
| `area: utils` | The narrowly owned utilities surface under `quantmind/utils/`. |
| `area: examples` | Examples are the primary deliverable, rather than supporting files for another area. |
| `area: packaging` | Installation, builds, dependencies, releases, and package metadata. |

Area labels describe ownership, not every touched path. For example, a news
design-document-only change is `type: docs` + `area: preprocess`, while a
skill-only guidance change is `type: docs` + `area: harness`.

## Impact Labels

| Label | Use when |
|---|---|
| `impact: breaking` | The accepted change can break a public API, serialized contract, or supported compatibility boundary. |
| `impact: live-network` | The work depends on or changes real public-network behavior, external availability, or a live smoke test. |

## Resolution and Community Labels

Keep these standard GitHub labels when applicable: `duplicate`,
`good first issue`, `help wanted`, `invalid`, `question`, and `wontfix`.
They describe resolution or collaboration state and are independent of the
type, area, and impact dimensions.

## Issues and Pull Requests

Body source formatting follows the [GitHub writing style](github-writing.md),
including its no-hard-wrap rule.

- Label an issue from the intended problem and accepted scope.
- Label a pull request from its actual diff.
- A pull request that closes an issue should normally align with the issue's
  type and area. If implementation differs, label the pull request accurately
  and update the issue when its accepted scope changed.
- Re-evaluate labels when scope changes; do not preserve a stale label merely
  because it was applied first.

## Examples

| Work | Labels | Reason |
|---|---|---|
| Context framework in [#101](https://github.com/LLMQuant/quant-mind/issues/101) | `type: feature`, `area: contexts` | It added a new routing capability. |
| CI/E2E consolidation in [#102](https://github.com/LLMQuant/quant-mind/issues/102) | `type: refactor`, `area: harness`, `impact: live-network` | It restructures repository verification, including live smoke tests. |
| This label-system issue and its PR | `type: docs`, `area: contexts`, `area: harness` | The canonical deliverable is context guidance routed through contributor controls. |
| Fix `paper_flow` contract behavior | `type: bug`, `area: flows` | An existing public operation is broken. |
| Update only the news design document | `type: docs`, `area: preprocess` | The document belongs to the news preprocessing surface, not contexts. |
| Add a public news source | `type: feature`, `area: preprocess`, `impact: live-network` | It adds acquisition behavior backed by a real external source. |

## Existing Label Migration

Rename labels instead of deleting and recreating them so GitHub preserves
history and current assignments:

| Existing | Rename to |
|---|---|
| `bug` | `type: bug` |
| `enhancement` | `type: feature` |
| `documentation` | `type: docs` |
| `refactor` | `type: refactor` |
| `example` | `area: examples` |
| `harness` | `area: harness` |

After renaming, create the missing taxonomy labels and audit open issues and
pull requests. Remove accidental multiple `type:` labels, choose one or two
areas from accepted scope, and leave resolution/community labels intact.
