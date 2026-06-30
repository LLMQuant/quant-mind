
<p align="center">
  <img src="assets/quantmind-new-orange-shaved.png" width="240">
</p>

<p align="center">
  <img src="assets/quant-mind.png" width="400">
</p>

<p align="center">
  <b>Turn Unstructured Documents into Durable, Agent-Ready Knowledge</b>
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

**QuantMind** is a finance-proven, agent-centric knowledge extraction library
built on top of the OpenAI Agents SDK. It gives AI agents better eyes over
PDF/HTML/text inputs, typed knowledge instead of brittle strings, and a clean
path toward durable memory and loop-based workflows.

Today, the first production flow is focused on quantitative-finance research.
But the architecture is intentionally broader: the same preprocess → typed
config → typed knowledge → agentic workflow stack can be reused for any
document-heavy domain.

### 📰 News
| 🗞️ News        | 📝 Description                                                                 |
|----------------|-------------------------------------------------------------------------------|
| 🎉 Accepted at NeurIPS 2025 Workshop | Our paper **[Quant-Mind](#)** has been accepted to the **[NeurIPS 2025 GenAI in Finance Workshop](https://sites.google.com/view/neurips-25-gen-ai-in-finance/home)** !🚀 |
| 📢 First Release on GitHub  | **Quant-Mind** is now live on GitHub — please check it out and join us! 🤗 |

### 🧐 Overview

QuantMind is designed for teams building **serious AI-agent workflows** around
documents, research, and structured memory.

Its current strength is finance-heavy research extraction, but its core value is
more general:

- fetch or accept raw source material from the web, local files, or inline text
- normalize it into markdown that an agent can reliably read
- force output into strict Pydantic knowledge objects
- preserve provenance so downstream agents can review, cite, and reuse results
- support repeatable loops instead of one-off prompt chains

### ✨ Why QuantMind?

#### For AI agents

- **Better eyes**: `preprocess/` turns PDFs, HTML, and raw text into stable
  markdown before the model sees them.
- **Better shared language**: `configs/` and `knowledge/` replace ad-hoc JSON
  with typed inputs, typed outputs, and explicit provenance.
- **Better loops**: `magic.py`, `flows/`, and `batch_run()` help agents resolve
  intent, execute work, and repeat tasks predictably.
- **Better long-term direction**: the `mind/` roadmap is explicitly about
  durable memory and agent-friendly retrieval, not throwaway prompting.

#### For domain teams

- **Finance-first, not finance-only**: the first production flow is optimized
  for research papers in quant finance, but the layering is reusable for any
  domain that needs document ingestion, typed extraction, and agentic reuse.
- **Strict architecture**: dependency boundaries are enforced with
  `import-linter`, and the repo ships with a single canonical verification loop.
- **Composable by design**: customization happens at three levels—config,
  flow kwargs, or forking a flow file.

---

### System Architecture

![quantmind-outline](assets/quantmind-stage-outline.png)

QuantMind is built on a decoupled architecture that separates source handling,
typed extraction, and future memory/store layers.

#### **Current production path**

```text
Natural-language intent
    ↓
magic.resolve_magic_input(...)
    ↓
typed input + typed cfg
    ↓
preprocess.fetch + preprocess.format
    ↓
flow(agent=OpenAI Agents SDK)
    ↓
typed knowledge object with provenance
```

#### **Permanent modules**

- `quantmind/flows/` — apex layer (`paper_flow`, `batch_run`, observability)
- `quantmind/configs/` — typed inputs and flow configuration
- `quantmind/knowledge/` — typed knowledge shapes and provenance contracts
- `quantmind/preprocess/` — fetch + format + cleaning helpers
- `quantmind/magic.py` — natural language to typed `(input, cfg)` resolution
- `quantmind/mind/` — reserved for the upcoming memory/store layer

See [docs/ARCHITECTURE_FOR_NEW_DOMAINS.md](docs/ARCHITECTURE_FOR_NEW_DOMAINS.md)
for how to extend this stack beyond finance.

---

### 🚀 Quick Start

We use [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

**Prerequisites:**

- Python 3.10+
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

### 📚 Usage Examples

#### Run a single finance paper through `paper_flow`

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

#### Use the same pipeline on a local memo or technical brief

```python
import asyncio
from pathlib import Path

from quantmind.configs import PaperFlowCfg
from quantmind.configs.paper import LocalFilePath
from quantmind.flows import paper_flow


async def main() -> None:
    doc = await paper_flow(
        LocalFilePath(path=Path("docs/internal-research-note.md")),
        cfg=PaperFlowCfg(model="gpt-4o-mini"),
    )
    print(doc.root.summary)


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

### 🔁 Agentic loops and durable memory

QuantMind is being tuned for workflows where multiple agents can understand one
another through **shared typed artifacts**, not just prompt conventions.

Today, that means:

- resolving loose intent into strict inputs with `magic.resolve_magic_input()`
- extracting typed knowledge with `paper_flow`
- scaling stateless fan-out work with `batch_run()`
- preserving provenance for review and downstream reuse

Next, that means:

- filesystem-backed working memory under `mind/memory`
- a store layer for retrieval and longer-lived agent loops
- stronger multi-step patterns for review, refinement, and replay

If you are building agent teams, think of QuantMind as the layer that provides
stable inputs, stable outputs, and a durable path from observation to memory.

---

### 🗺️ Roadmap

- [x] Remove the legacy in-repo agent runtime
- [x] Reposition QuantMind on top of OpenAI Agents SDK
- [x] Land the permanent module roots: `flows/`, `configs/`, `knowledge/`,
  `preprocess/`, `magic.py`
- [x] Ship a canonical verification loop (`scripts/verify.sh`)
- [ ] Add filesystem-backed working memory in `mind/memory`
- [ ] Add the store/retrieval layer in `mind/store`
- [ ] Expand beyond the first paper flow with more domain flows
- [ ] Improve multi-agent loop patterns, observability, and replay support
- [ ] Keep agent-facing docs and extension paths under regular review during
  active weekly iteration

---

### The Vision: An Intelligent Research Framework

> [!IMPORTANT]
> **This section describes our long-term vision, not current capabilities.**
> QuantMind already ships a useful extraction stack, but the broader memory and
> retrieval story is still under construction.

QuantMind is being shaped into a durable intelligence layer for document-heavy
workflows. Finance remains a major proving ground, but the long-term value is
larger: helping AI agents see source material clearly, preserve structured
understanding over time, and operate in loops that do not lose context between
runs.

The near-term roadmap starts with papers and research-heavy workflows. The
longer-term goal is a reusable foundation for agentic knowledge work across
domains.

> [!NOTE]
> **Future Conceptual Example (PR6 brings `FilesystemMemory`):**
>
> ```python
> from quantmind.configs.paper import ArxivIdentifier
> from quantmind.flows import paper_flow
> from quantmind.knowledge import Paper
> from quantmind.mind.memory import FilesystemMemory  # PR6
>
> memory = FilesystemMemory("./mem/factor-research/")
> for arxiv_id in arxiv_ids:
>     paper: Paper = await paper_flow(ArxivIdentifier(id=arxiv_id), memory=memory)
> ```

This future state represents the shift from one-off extraction to reusable,
memory-aware, agentic knowledge systems.

------

### 🤝 Contributing

We welcome contributions of all forms, from bug reports to feature development.

> [!IMPORTANT]
> **For Contributors**: Please read [CONTRIBUTING.md](CONTRIBUTING.md) for essential development setup including pre-commit hooks, coding standards, and testing requirements.

**Quick Start for Contributors:**

1. **Fork** the repository
2. **Setup the development environment**:

   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e ".[dev]"
   ./scripts/pre-commit-setup.sh
   ```

3. **Create feature branch** (`git checkout -b feat/my-feature`)
4. **Follow conventional commits** (`feat: add new feature`)
5. **Submit PR** with our template

**Before Contributing:**

- Open an [issue](https://github.com/LLMQuant/quant-mind/issues) to discuss significant changes
- Use our issue templates for bug reports and feature requests
- Ensure `bash scripts/verify.sh` passes before submitting PR

### License

QuantMind is released under the MIT License—see `LICENSE` for details.

### ❤️ Acknowledgements

- **arXiv** for providing open access to a world of research.
- The **open-source community** for the tools and libraries that make this project possible.
