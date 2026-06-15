# Fincept ↔ QuantMind Integration Guide

## Overview

QuantMind serves as the **knowledge extraction layer** for Fincept AI Ops.
Rather than raw market data, Fincept can query QuantMind for:
- Distilled research insights from financial papers
- Structured signal metadata (alpha factors, risk signals)
- Natural language Q&A over quantitative finance literature

## Integration Architecture

```
fincept-ai-ops (FastAPI)
    │
    ├── GET /quant/search?q={query}
    │       ↓
    │   QuantMind Knowledge API
    │       ↓
    │   Vector Search (LanceDB / Chroma)
    │       ↓
    │   Extracted Insights JSON
    │
    └── MCP Tool: query_quantmind
```

## Usage Example

```python
from quantmind import QuantMindClient

client = QuantMindClient(api_key="...")

# Search for momentum factor research
results = client.search(
    query="momentum factor decay regime change",
    top_k=5,
    filters={"source_type": "paper", "year_gte": 2020}
)

for r in results:
    print(r.title, r.insight_summary, r.confidence_score)
```

## Synergy with Fincept

| Fincept Component | QuantMind Feature |
|---|---|
| Strategy generation | Research paper extraction |
| Risk model inputs | Factor library |
| Audit log enrichment | Source citation tracking |
| Backtest hypothesis | Literature-validated signals |

## Setup

1. Clone both repos side-by-side
2. Set `QUANTMIND_API_URL` in fincept `.env`
3. Run `quantmind serve --port 8001`
4. Fincept auto-discovers at startup

See [fincept-ai-ops ROADMAP](../fincept-ai-ops/ROADMAP.md) for v1.2 integration milestone.
