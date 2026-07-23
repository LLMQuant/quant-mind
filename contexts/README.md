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

This directory is the public repository context system for coding agents and maintainers. For a full page-by-page navigation index, see [`CONTEXT_MAP.md`](CONTEXT_MAP.md). Start with the index that matches the work you are doing:

- [Develop QuantMind](dev/README.md) for architecture, contribution, testing, and verification guidance.
- [Use QuantMind](usage/README.md) for public operations, examples, and usage documentation.
- [Design QuantMind](design/README.md) for accepted design decisions and planned behavior that spans packages.

## Where Information Lives

- **`contexts/` vs `docs/`**: `contexts/` is the internal navigation and design memory for coding agents and maintainers — design decisions in `contexts/design/`, contributor rules in `contexts/dev/`, usage routing in `contexts/usage/`. `docs/` is the user-facing surface for people consuming QuantMind as a library — the public component catalog `docs/README.md` and usage guides such as `docs/library.md`. A fact that helps an agent build the repo lives in `contexts/`; a fact that helps someone use the library lives in `docs/`. Neither keeps a second copy of the other.
- Keep each kind of information in one place so agents do not have to compare competing versions.
- `contexts/design/` records accepted design decisions and planned behavior.
- `contexts/dev/` and `contexts/usage/` route readers to existing rules and examples instead of copying them.
- Code and tests show current behavior. Design pages list any planned behavior that is not implemented yet.
- `docs/` remains the home for user-facing guides, examples, and catalogs. It may link to a design page but must not maintain a second copy of the design.
- Update a component's context index only when its discoverable entry points change, not for every implementation change.
- Repository maintainers own this shared structure. Component contributors own the links for the components they change.

Keep this layer limited to public repository information. Private, deployment-specific, and internal-only guidance does not belong here.
