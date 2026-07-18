# GRPO Suitability Tagging

## Overview

Each QuantMind corpus entry is tagged with a `grpo_suitability` field
(`"high"`, `"medium"`, or `"low"`) at ingest time. The tag scores how
useful the entry would be as training or evaluation data for a
Generalized Reward Policy Optimization (GRPO) loop, specifically whether
the entry's content sits in the **learnable zone** — questions it encodes
can be answered by a strong solver but not a weak one.

## Theoretical Basis

The framework is grounded in **Autodata** (Kulikov, Whitehouse, Wu, Nie
et al., FAIR at Meta, arXiv:2606.25996, 2026). Section 2b defines the
acceptance criterion for a GRPO-useful training example:

> strong solver avg ≥ 0.65, weak solver avg < 0.50, gap ≥ 20pp

Content that meets this criterion sits in the "learnable zone": too hard
for a weak model to recall from surface features, but consistently
solvable by a capable model using deep reasoning. Content outside this
zone either provides no discrimination signal (both solvers fail — too
hard) or no learning signal (both solvers succeed — too easy).

Autodata's empirical result on legal reasoning tasks: 4.8% high-suitability
entries with naive CoT generation → 52% high-suitability entries after the
agentic loop. The gap shows how much corpus quality varies and why tagging
matters before training or evaluation.

## V1 Heuristic (No Live Model Calls)

V1 uses a deterministic heuristic on fields already present in the corpus
entry. No network calls, no LLM inference. The heuristic uses three signals
as proxies for discrimination potential:

| Signal | Proxy for |
|---|---|
| `length_band` | Document depth (more content → richer reasoning surface) |
| `domain_band` | Source authority (peer-reviewed → higher reasoning demand) |
| `code_present` | Technical depth (math/code → non-trivial query surface) |

### Length Band

| Band | Threshold |
|---|---|
| `short` | `markdown_chars` < 5 000 |
| `medium` | 5 000 ≤ `markdown_chars` < 20 000 |
| `long` | `markdown_chars` ≥ 20 000 |

### Domain Band

| Band | Source types / URL patterns |
|---|---|
| `arxiv` | `source_type == "arxiv"` or `source_type == "local"` or URL contains `arxiv.org` |
| `ssrn` | URL contains `ssrn.com` |
| `substack` | URL contains `substack.com` |
| `news` | All other URLs, `source_type == "text"`, unrecognized sources |

### Code Present

`True` if the entry's markdown contains a fenced code block (` ``` `),
a math block (`$$`), or inline code of ≥ 4 characters.

### V1 Decision Rule

```
long + arxiv + code_present        → "high"
short + news + not code_present    → "low"
everything else                    → "medium"
```

The rule is conservative: only the clearest signals on both ends are
tagged high or low. Uncertain cases default to medium.

## Backward Compatibility

Existing corpus entries that pre-date this feature simply lack the
`grpo_suitability` key. The scorer operates on any dict and returns a
score regardless of which optional fields are present — it will not raise
on a partial or legacy record. Downstream consumers should treat a missing
key as `null` (unscored), not as `"low"`.

## Schema Impact

### `~/.quantmind/corpus/items/<id>.json`

```jsonc
{
  // ... existing fields unchanged ...
  "grpo_suitability": "high"   // "high" | "medium" | "low"
}
```

### `~/.quantmind/corpus/ingestion_log.jsonl`

```jsonc
{
  "id": "...",
  "title": "...",
  "source_type": "arxiv",
  "source": "2606.25996",
  "ingested_at": "...",
  "grpo_suitability": "high",
  "event": "research.ingest"
}
```

## V2 Plan — Actual Solver Gap

When QuantMind has the LLM substrate to run two queries per entry, replace
the heuristic with a real discrimination measurement:

1. **Weak query** — surface-recall question: *"What method did this paper
   propose?"* Run via a cheap model (Haiku / small LLAMA).
2. **Strong query** — application question: *"Where does this method break
   down in a non-stationary regime?"* Run via a capable model (Sonnet /
   Opus).
3. **Gap** = strong score − weak score.
4. Tag `high` if `gap ≥ 0.20` AND `strong ≥ 0.65`; `low` if `gap < 0.05`
   AND `strong < 0.50`; `medium` otherwise.

The scorer class (`GrpoSuitabilityScorer`) already carries documented TODO
hooks in `qm_mcp/grpo_suitability.py` marking where these steps plug in.

## Usage

The tag is computed automatically at ingest time. To query by suitability,
filter the corpus store items by the `grpo_suitability` field. Priority
for Conductor and Strategy Lab use cases: surface `"high"` entries first.
