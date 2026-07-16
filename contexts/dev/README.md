# Develop QuantMind

Use this index when extending or maintaining the repository. Follow the linked
canonical source instead of treating this page as a replacement for it.

| Need | Canonical source |
|---|---|
| Architecture constraints and module ownership | [`AGENTS.md`](../../AGENTS.md) or [`CLAUDE.md`](../../CLAUDE.md) |
| Component-development workflow | [`quantmind-dev` component workflow](../../.agents/skills/quantmind-dev/references/develop-components.md) |
| Issue and pull-request labels | [Repository label guidance](labels.md) |
| Issue and pull-request body formatting | [GitHub writing style](github-writing.md) |
| Public operation and source catalog | [`docs/README.md`](../../docs/README.md) |
| Public operation naming | [Operation naming contract](../design/operations/naming.md) |
| Component designs | [Canonical design index](../design/README.md) |
| Test patterns and coverage | [`tests/`](../../tests/) |
| Focused runnable examples | [`examples/`](../../examples/) |
| Deterministic verification | [`scripts/verify.sh`](../../scripts/verify.sh) |

Run `bash scripts/verify.sh` for every change. For a public-network component,
also run the bounded live verifier listed in the public component catalog.
