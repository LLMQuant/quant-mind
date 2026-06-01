
<p align="center">
  <img src="assets/quantmind-new-orange-shaved.png" width="240">
</p>

<p align="center">
  <img src="assets/quant-mind.png" width="400">
</p>

<p align="center">
  <b>Transform Financial Knowledge into Actionable Intelligence</b>
</p>
<p align="center">
  <a href="https://github.com/LLMQuant/quant-mind/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  </a>
</p>
<p align="center">
  <a href="#-why-quantmind">Why QuantMind</a> •
  <a href="#system-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-usage-examples">Usage</a> •
  <a href="#%EF%B8%8F-roadmap">Roadmap</a> •
  <a href="#the-vision-an-intelligent-research-agent">Vision</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

**QuantMind** is an intelligent knowledge extraction and retrieval framework for quantitative finance. It transforms unstructured financial content—papers, news, blogs, reports—into a queryable knowledge base, enabling AI-powered research at scale.

### 📰 News
| 🗞️ News        | 📝 Description                                                                 |
|----------------|-------------------------------------------------------------------------------|
| 🎉 Accepted at NeurIPS 2025 Workshop | Our paper **[Quant-Mind](#)** has been accepted to the **[NeurIPS 2025 GenAI in Finance Workshop](https://sites.google.com/view/neurips-25-gen-ai-in-finance/home)** !🚀 |
| 📢 First Release on GitHub  | **Quant-Mind** is now live on GitHub — please check it out and join us! 🤗 |

### 🧐 Overview

QuantMind is a next-generation AI platform that ingests, processes, and structures **every** new piece of quantitative-finance research, including papers, news, blogs, and SEC filings into a **semantic knowledge graph**. Institutional investors, hedge funds, and research teams can now explore the frontier of factor strategies, risk models, and market insights in **seconds**, unlocking alpha that would otherwise remain buried.

### ✨ Why QuantMind?

The financial research landscape is overwhelming. Every day, hundreds of papers, articles, and reports are published.

#### 🌐 The Opportunity

- **Information Overload**: 500 new research papers & reports published daily. Manual review takes weeks—costly, error-prone, and non-scalable
- **Massive Market**: Financial data & analytics market ≫ expected to grow to US$961.89 billion by 2032, with a compound annual growth rate of 13.5%. Tens of thousands of quant teams & asset managers hungry for speed
- **High ROI**: 1% improvement in research efficiency can translate to millions saved or earned in trading performance

---

#### 💡 **QuantMind** solves this by

- 🔍 **Extracting** structured knowledge from any source (PDFs, web pages, APIs)
- 🧠 **Understanding** content with domain-specific LLMs fine-tuned for finance
- 💾 **Storing** information in a semantic knowledge graph
- 🚀 **Retrieving** insights through natural language queries

---

### System Architecture

![quantmind-outline](assets/quantmind-stage-outline.png)

QuantMind is built on a decoupled, two-stage architecture. This design separates the concerns of data ingestion from intelligent retrieval, ensuring both robustness and flexibility.

#### **Stage 1: Knowledge Extraction**

This layer is responsible for collecting, parsing, and structuring raw information into standardized knowledge units.

```text
Source APIs (arXiv, News, Blogs) → Intelligent Parser → Workflow/Agent → Structured Knowledge Base
```

- **Source**: Connects to various sources (academic APIs, news feeds, financial blogs, perplexity search source) to pull content
- **Parser**: Extracts text, tables, and figures from PDFs, HTML, and other formats
- **Tagger**: Automatically categorizes content into research areas and topics
- **Workflow/Agent**: Orchestrates the extraction pipeline with quality control and deduplication

#### **Stage 2: Intelligent Retrieval**

This layer transforms structured knowledge into actionable insights through various retrieval mechanisms.

```
Knowledge Base → Embeddings → Solution Scenarios (DeepResearch, RAG, Data MCP, ...)
```

- **Embedding Generation**: Converts knowledge units into high-dimensional vectors for semantic search

- Solution Scenarios: Multiple retrieval patterns including:

  - **DeepResearch**: Complex multi-hop reasoning across documents
  - **RAG**: Retrieval-augmented generation for Q&A
  - **Data MCP**: Structured data access protocols
  - Custom retrieval patterns based on use case

---

### 🚀 Quick Start

We use [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

**Prerequisites:**

- Python 3.8+
- Git

**Installation:**

1. **Install uv (if not already installed):**

   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   # Or using pip
   pip install uv
   ```

2. **Clone the repository:**

   ```bash
   git clone https://github.com/LLMQuant/quant-mind.git
   cd quant-mind
   ```

3. **Create and activate virtual environment:**

   ```bash
   # Create a virtual environment
   uv venv

   # Activate it
   # On macOS/Linux:
   source .venv/bin/activate

   # On Windows:
   .venv\Scripts\activate
   ```

4. **Install dependencies:**

   ```bash
   uv pip install -e .
   ```

5. **Optional — Install Node.js + `npx` (only if you plan to use `FilesystemMemory`):**

   `FilesystemMemory` launches the MCP filesystem server through `npx`,
   so cross-step memory examples need a Node.js toolchain. Skip this if
   you do not call `paper_flow(..., memory=FilesystemMemory(...))`.

   ```bash
   # macOS (Homebrew)
   brew install node

   # Linux (Debian/Ubuntu)
   curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
   sudo apt-get install -y nodejs

   # Verify
   node -v       # >= 18.x recommended
   npx --version
   ```

   The first run of `npx -y @modelcontextprotocol/server-filesystem`
   downloads the package into the local npm cache; subsequent runs
   reuse it.

### 📚 Usage Examples

#### Run a single paper through `paper_flow`

```python
import asyncio

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import ArxivIdentifier
from quantmind.flows import paper_flow


async def main() -> None:
    paper = await paper_flow(
        ArxivIdentifier(id="2401.12345"),
        cfg=PaperFlowCfg(model="gpt-4o-mini"),
    )
    print(paper.model_dump_json(indent=2))


asyncio.run(main())
```

#### Fan out a batch with `batch_run`

```python
import asyncio

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import ArxivIdentifier
from quantmind.flows import batch_run, paper_flow


async def main() -> None:
    inputs = [ArxivIdentifier(id=aid) for aid in (
        "2401.12345", "2401.12346", "2401.12347",
    )]
    result = await batch_run(
        paper_flow,
        inputs,
        cfg=PaperFlowCfg(model="gpt-4o-mini"),
        concurrency=3,
        on_error="skip",
        on_progress=lambda done, total: print(f"{done}/{total}"),
    )
    print(f"ok={result.success_count} failed={result.failure_count}")


asyncio.run(main())
```

#### Resolve free-form intent with `magic`

```python
import asyncio

from quantmind.flows import paper_flow
from quantmind.magic import resolve_magic_input


async def main() -> None:
    inp, cfg = await resolve_magic_input(
        "Pull arXiv 2401.12345 about cross-sectional momentum; use gpt-4o-mini.",
        target_flow=paper_flow,
    )
    paper = await paper_flow(inp, cfg=cfg)
    print(paper.model_dump_json(indent=2))


asyncio.run(main())
```

#### Persistent memory across a serial loop

```python
import asyncio

from quantmind.configs.paper import ArxivIdentifier
from quantmind.flows import paper_flow
from quantmind.mind.memory import FilesystemMemory


async def main() -> None:
    mem = FilesystemMemory("./.qm-memory")
    arxiv_ids = ["1706.03762", "1810.04805", "2005.11401"]
    for paper_id in arxiv_ids:
        paper = await paper_flow(
            ArxivIdentifier(id=paper_id),
            memory=mem,
        )
        print(paper.title)
    # Trajectory records are now under ./.qm-memory/runs/.


asyncio.run(main())
```

`FilesystemMemory` requires Node.js + `npx` on PATH (the SDK launches
`@modelcontextprotocol/server-filesystem` over stdio). The Agent sees
`<memory_dir>/workspace/` (`notes/` and `items/`); system trajectory
records stay outside that MCP root under `<memory_dir>/runs/<run_id>.json`
and `<memory_dir>/runs.jsonl`. `FilesystemMemory` is for serial loops
only — `batch_run` rejects `memory=` at the signature layer.

> **Note**: QuantMind is mid-migration to OpenAI Agents SDK
> (see [#71](https://github.com/LLMQuant/quant-mind/issues/71)). PR6 lands
> `mind/memory/` (Memory Protocol + `FilesystemMemory` MVP + trajectory
> archive); the remaining work is the `mind/store/` knowledge layer
> scheduled for PR7+.

---

### 🗺️ Roadmap

- [x] Better `flow` design for user-friendly usage
- [x] First production level example (Quant Paper Agent)
- [x] Migrate Agent layer to OpenAI Agents SDK
- [x] Standardize knowledge format with `knowledge/` (Pydantic-based)
- [x] Cross-step working memory (`mind/memory`) for serial document processing
- [ ] Additional content sources (financial news, blogs, reports)
- [ ] `mind/store/` — durable knowledge store with hybrid retrieval (PR7+)

---

### The Vision: An Intelligent Research Framework

> [!IMPORTANT]
> **This section describes our long-term vision, not current capabilities.** While QuantMind today provides a solid knowledge extraction framework, the features described below represent our aspirational goals for future development.

QuantMind is designed with a larger vision: to become a comprehensive intelligence layer for all financial knowledge. We're building toward a system that understands the interconnections between academic research, market news, analyst reports, and social sentiment—creating a unified knowledge base that powers better financial decisions.

The foundation we're building today—starting with papers—will expand to encompass the entire financial information ecosystem.

> [!NOTE]
> **`FilesystemMemory` landed in PR6.** See the runbook example above
> (*Persistent memory across a serial loop*) for the canonical usage.
> Future cognitive layers (`mind/store`, `mind/summarize_run`) build on
> this foundation — they share the same `<memory_dir>/` layout.

This future state represents our commitment to moving beyond simple data aggregation and toward genuine machine intelligence in the financial domain.

------

### 🤝 Contributing

We welcome contributions of all forms, from bug reports to feature development.

> [!IMPORTANT]
> **For Contributors**: Please read [CONTRIBUTING.md](CONTRIBUTING.md) for essential development setup including pre-commit hooks, coding standards, and testing requirements.

**Quick Start for Contributors:**

1. **Fork** the repository
2. **Setup development environment**:

   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e .
   ./scripts/pre-commit-setup.sh
   ```

3. **Create feature branch** (`git checkout -b feat/my-feature`)
4. **Follow conventional commits** (`feat: add new feature`)
5. **Submit PR** with our template

**Before Contributing:**

- Open an [issue](https://github.com/LLMQuant/quant-mind/issues) to discuss significant changes
- Use our issue templates for bug reports and feature requests
- Ensure all pre-commit hooks pass before submitting PR

### License

QuantMind is released under the MIT License—see `LICENSE` for details.

### ❤️ Acknowledgements

- **arXiv** for providing open access to a world of research.
- The **open-source community** for the tools and libraries that make this project possible.
