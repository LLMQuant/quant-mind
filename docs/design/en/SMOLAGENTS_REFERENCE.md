# Vendored smolagents Reference (Archive)

This file is part of the `archive/agent-runtime-final` branch and records the
smolagents source snapshot that was used while building the self-built
agent runtime (PRs #61, #65, #67).

- Repository: https://github.com/huggingface/smolagents.git
- Commit: `c9cd01af3eaf9ff981bb33fd62e946d39b8bd5a8`
- Tag: `v1.0.0-824-gc9cd01a`

To reconstruct the exact vendored copy that lived at `smolagents/` in this branch:

```bash
git clone https://github.com/huggingface/smolagents.git
cd smolagents
git checkout c9cd01af3eaf9ff981bb33fd62e946d39b8bd5a8
```

The actual source tree is not committed here because it was a nested git
working copy, which would be stored as a gitlink rather than real content.
