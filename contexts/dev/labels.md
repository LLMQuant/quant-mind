# Repository Label Guidance

## Quick Summary

- **Purpose**: Select consistent labels for QuantMind issues and pull requests.
- **Read when**: Creating, updating, triaging, or reviewing an issue or PR.
- **Label count**: Use exactly one `type:` label, normally one `area:` label
  (at most two), and optional `impact:` labels.
- **Selection rule**: Label the work's actual purpose and main area, not every
  file it touches.

## Contents

- [How Many Labels to Use](#how-many-labels-to-use)
- [Type Labels](#type-labels)
- [Area Labels](#area-labels)
- [Impact Labels](#impact-labels)
- [Resolution and Community Labels](#resolution-and-community-labels)
- [Issues and Pull Requests](#issues-and-pull-requests)
- [Examples](#examples)
- [Rename Existing Labels](#rename-existing-labels)

Labels describe three separate parts of the work:

- `type:` answers what kind of change is proposed or delivered.
- `area:` answers which part of the repository owns the work.
- `impact:` identifies breaking changes or live-network risk.

Choose labels from the work's purpose and agreed scope, not only from the files
it happens to touch.

## How Many Labels to Use

Every issue and pull request must have **exactly one** `type:` label.

Normally choose **one** `area:` label. Use two only when the work genuinely
belongs to two main areas; never use more than two.

Add `impact:` labels only when needed. Resolution and community labels do not
count toward these limits.

## Type Labels

| Label | Use when |
|---|---|
| `type: bug` | Existing behavior does not match its documented or expected behavior. |
| `type: feature` | The work adds a new capability or observable behavior. |
| `type: refactor` | Existing implementation is reorganized without intentionally changing public behavior. |
| `type: docs` | The primary deliverable is documentation, guidance, or explanatory example text. |
| `type: maintenance` | The work performs specific repository upkeep, such as dependency, configuration, release, or housekeeping changes. |
| `type: design` | The primary deliverable is an architecture decision, RFC, or design that precedes implementation. |

`type: maintenance` is not a catch-all label. If the intended outcome is
unclear, clarify the work before labeling it.

## Area Labels

| Label | Main area |
|---|---|
| `area: contexts` | Repository information routing and discovery under `contexts/`. This does not mean all documentation. |
| `area: harness` | Contributor and agent controls: `AGENTS.md`, `CLAUDE.md`, skills, CI, verification scripts, hooks, templates, and rulesets. |
| `area: knowledge` | Knowledge models, stored formats, serialization, and representation under `quantmind/knowledge/`. |
| `area: configs` | Typed inputs and configuration models under `quantmind/configs/`. |
| `area: preprocess` | Fetching, parsing, cleaning, formatting, and source handling under `quantmind/preprocess/`. |
| `area: rag` | Opinionated document chunking, indexing, and retrieval under `quantmind/rag/`. |
| `area: flows` | Public operation implementations under `quantmind/flows/`. It is not a generic synonym for pipeline or orchestration. |
| `area: mind` | Memory, tools, MCP integration, and agent reasoning under `quantmind/mind/`. |
| `area: utils` | The narrowly owned utilities surface under `quantmind/utils/`. |
| `area: examples` | Examples are the primary deliverable, rather than supporting files for another area. |
| `area: packaging` | Installation, builds, dependencies, releases, and package metadata. |

Area labels describe the main work, not every touched path. For example, a news
design-document-only change is `type: docs` + `area: preprocess`, while a
skill-only guidance change is `type: docs` + `area: harness`.

## Impact Labels

| Label | Use when |
|---|---|
| `impact: breaking` | The accepted change can break a public API, stored data format, or supported integration. |
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
  and update the issue when its agreed scope changed.
- Re-evaluate labels when scope changes; do not preserve a stale label merely
  because it was applied first.

## Examples

| Work | Labels | Reason |
|---|---|---|
| Context framework in [#101](https://github.com/LLMQuant/quant-mind/issues/101) | `type: feature`, `area: contexts` | It added a new routing capability. |
| CI/E2E consolidation in [#102](https://github.com/LLMQuant/quant-mind/issues/102) | `type: refactor`, `area: harness`, `impact: live-network` | It restructures repository verification, including live smoke tests. |
| This label-system issue and its PR | `type: docs`, `area: contexts`, `area: harness` | The main result is context guidance routed through contributor controls. |
| Fix documented `paper_flow` behavior | `type: bug`, `area: flows` | An existing public operation is broken. |
| Update only the news design document | `type: docs`, `area: preprocess` | The document belongs to the news preprocessing surface, not contexts. |
| Add a public news source | `type: feature`, `area: preprocess`, `impact: live-network` | It adds acquisition behavior backed by a real external source. |

## Rename Existing Labels

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

After renaming, create the missing labels and audit open issues and
pull requests. Remove accidental multiple `type:` labels, choose one or two
areas from accepted scope, and leave resolution/community labels intact.
