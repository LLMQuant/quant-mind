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

    import asyncio
    from pathlib import Path

    from quantmind.configs import PaperFlowCfg
    from quantmind.configs.paper import LocalFilePath
    from quantmind.flows import paper_flow


    async def main() -> None:
        doc = await paper_flow(
            LocalFilePath(path=Path("path/to/document.pdf")),
            cfg=PaperFlowCfg(model="gpt-4o-mini"),
        )
        # Persist / index this JSON in Fincept's storage layer.
        print(doc.model_dump(mode="json"))


    asyncio.run(main())

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
