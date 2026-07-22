# QuantMind Repository Context

## Quick Summary

- **Purpose**: Route agents to the smallest set of context pages needed for a QuantMind task.
- **Read when**: Starting repository development, library usage, or design work.
- **Load next**: Choose exactly one primary route below, then follow only links required by the task.
- **Authority**: Design pages record agreed behavior; code and tests show what the current version implements.

## Contents

- [Routes](#routes)
- [Where Information Lives](#where-information-lives)

## Routes

This directory is the public repository context system for coding agents and maintainers. Start with the index that matches the work you are doing:

- [Develop QuantMind](dev/README.md) for architecture, contribution, testing, and verification guidance.
- [Use QuantMind](usage/README.md) for public operations, examples, and usage documentation.
- [Design QuantMind](design/README.md) for accepted design decisions and planned behavior that spans packages.

## Where Information Lives

- Keep each kind of information in one place so agents do not have to compare competing versions.
- `contexts/design/` records accepted design decisions and planned behavior.
- `contexts/dev/` and `contexts/usage/` route readers to existing rules and examples instead of copying them.
- Code and tests show current behavior. Design pages list any planned behavior that is not implemented yet.
- `docs/` remains the home for user-facing guides, examples, and catalogs. It may link to a design page but must not maintain a second copy of the design.
- Update a component's context index only when its discoverable entry points change, not for every implementation change.
- Repository maintainers own this shared structure. Component contributors own the links for the components they change.

Keep this layer limited to public repository information. Private, deployment-specific, and internal-only guidance does not belong here.
