# Fincept ↔ QuantMind Integration Guide

## Overview

QuantMind serves as the **knowledge extraction layer** for Fincept AI Ops.
Rather than raw market data, Fincept can call QuantMind to:

- Distill research insights from financial papers
- Produce structured signal metadata (alpha factors, risk signals)
- Answer natural-language questions over quantitative finance literature

## Integration Architecture

QuantMind is consumed from Python (no dedicated HTTP service ships in this
repo). Fincept can call QuantMind flows directly from its worker process, or
wrap them in a thin internal endpoint if a network boundary is required.

```
fincept-ai-ops (worker / FastAPI endpoint)
    |
    | from quantmind.flows import paper_flow
    | doc = await paper_flow(LocalFilePath(...), cfg=PaperFlowCfg(...))
    v
QuantMind Flow (paper_flow)
    |
    v
Extracted Insights JSON (typed Pydantic + provenance)
```

## Usage Example

```python
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
```

## Synergy with Fincept

| Fincept Component        | QuantMind Feature                          |
|--------------------------|--------------------------------------------|
| Strategy generation      | Research paper extraction                  |
| Risk model inputs        | Factor library                             |
| Audit log enrichment     | Source citation tracking                   |
| Backtest hypothesis      | Literature-validated signals               |

## Setup

1. Install QuantMind alongside Fincept in the same Python environment
   (`pip install -e .` from this repository).
2. From Fincept, import the flow you need (for example `paper_flow`) and
   call it as shown in the usage example. No `quantmind serve` CLI ships
   with this repo; if Fincept needs a network boundary, expose `paper_flow`
   (or any other flow) through your own thin FastAPI wrapper.

For broader context on the Fincept integration milestone, see the Fincept
project's own roadmap in the Fincept repository.
