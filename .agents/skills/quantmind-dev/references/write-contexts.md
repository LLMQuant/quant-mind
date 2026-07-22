# Writing contexts pages for QuantMind

The canonical standard for authoring or editing anything under `contexts/`. Read
it before adding or changing a design or dev page. `tests/test_contexts.py`
(part of `scripts/verify.sh`) enforces the structural rules below, so a page
that ignores them fails the build.

## When to add a page

- Add a design page only for real design content — an agreed behavior an
  implementation must preserve. No placeholders, empty directories, or
  speculative component pages.
- Organize by package or feature: `contexts/design/<package>/<topic>.md`. Do not
  add an intermediate `components/` directory.
- Keep code the source of truth for behavior; a page explains decisions and
  lists any current gap (state whether it is current, planned, or both).

## Required page structure

- Every `contexts/**/*.md` opens with `## Quick Summary`, then `## Contents`,
  both within the first 80 lines and in that order.
- Quick Summary is the routing preview (Purpose / Read when / Load next or Owner
  / Status). It routes a reader; it does not replace reading the page in full.
- The `## Contents` links must **exactly** match the page's `##` section
  headings, excluding Quick Summary and Contents. Anchors are GitHub-style:
  lowercase, punctuation stripped, spaces and hyphens collapsed to one hyphen
  (`## The Fallback Ladder` → `#the-fallback-ladder`).
- Register the page in its index and keep index links resolving: a design page
  in the global index `contexts/design/README.md`; a dev route in
  `contexts/dev/README.md`.
- Match the prose wrapping of the surrounding pages (existing contexts pages
  hard-wrap at roughly 76 columns). The no-hard-wrap rule in `github-writing.md`
  is for GitHub issue and PR bodies, not for contexts pages.

## Mermaid diagrams

GitHub renders a fenced ```` ```mermaid ```` block natively — embed the diagram,
never a pre-rendered image. A mermaid block adds no `##` heading, so it never
affects the Quick Summary / Contents anchor check. Keep diagrams compact:

- **Direction follows the diagram's story, not the screen.** Use `LR` for a
  linear pipeline, ladder, or request path; use `TD` for a layered or
  hierarchical structure (a subgraph-per-layer map).
- **Prefer `LR` whenever the flow is essentially linear.** Mermaid's default
  node height is large, so a `TD` chart stacks tall and pushes the page down
  several screens; GitHub instead scrolls a wide `LR` chart horizontally.
- **Short node text.** Push detail to edge labels or the prose beneath the
  diagram. Quote any label with special characters (`()`, `/`, `=`, `→`).
- **Split past ~15–20 nodes**, or group with `subgraph`. Do not mix `TD` and
  `LR` in one chart; use subgraphs if parts need different layouts.

## Required check

`tests/test_contexts.py` runs inside `bash scripts/verify.sh` and validates the
progressive-disclosure structure, the Contents/anchor match, and every index
link. Run it before pushing a contexts change.
