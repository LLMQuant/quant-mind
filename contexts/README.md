# QuantMind Repository Context

This directory is the public repository context system for coding agents and
maintainers. Start with the index that matches the work you are doing:

- [Develop QuantMind](dev/README.md) for architecture, contribution, testing,
  and verification guidance.
- [Use QuantMind](usage/README.md) for public operations, examples, and usage
  documentation.
- [Design QuantMind](design/README.md) for canonical engineering designs and
  cross-domain target contracts.

## Source of Truth and Ownership

- `contexts/design/` is the canonical source for engineering design decisions
  and target contracts.
- `contexts/dev/` and `contexts/usage/` curate links and route readers; they do
  not copy full guidance.
- Code and tests remain authoritative for current runtime behavior. Design
  pages identify current gaps when they describe behavior that has not landed.
- `docs/` remains the home for user-facing guides, examples, and catalogs. It
  may link to canonical design context but must not maintain a second copy.
- Update a component's context index only when its discoverable entry points
  change, not for every implementation change.
- Repository maintainers own this shared structure. Component contributors own
  the links for the components they change.

Keep this layer limited to public repository information. Private,
deployment-specific, and internal-only guidance does not belong here.
