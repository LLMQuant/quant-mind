# Develop QuantMind

## Quick Summary

- **Purpose**: Route contributors and coding agents to existing development
  rules without copying them here.
- **Read when**: Extending, fixing, reviewing, testing, or publishing QuantMind.
- **Load next**: Select the single row that matches the task; load additional
  sources only when that workflow explicitly requires them.
- **Required check**: Run `bash scripts/verify.sh` for every repository change.

## Contents

- [Rule Index](#rule-index)
- [Verification Rule](#verification-rule)

## Rule Index

Use this index when extending or maintaining the repository. Follow the linked
rule instead of treating this page as a replacement for it.

| Need | Read |
|---|---|
| Package responsibilities and coding rules | [`AGENTS.md`](../../AGENTS.md) or [`CLAUDE.md`](../../CLAUDE.md) |
| Component-development workflow | [`quantmind-dev` component workflow](../../.agents/skills/quantmind-dev/references/develop-components.md) |
| Issue and pull-request labels | [Repository label guidance](labels.md) |
| Issue and pull-request body formatting | [GitHub writing style](github-writing.md) |
| Public operation and source catalog | [`docs/README.md`](../../docs/README.md) |
| Public operation naming | [Operation naming rules](../design/operations/naming.md) |
| Component designs | [Design index](../design/README.md) |
| How to write tests (standards + change→test obligations) | [Writing tests](../../.agents/skills/quantmind-dev/references/tests.md) |
| Focused runnable examples | [`examples/`](../../examples/) |
| All local checks | [`scripts/verify.sh`](../../scripts/verify.sh) |

## Verification Rule

Run `bash scripts/verify.sh` for every change. For a public-network component,
also run the limited live-network check listed in the public component
catalog.
