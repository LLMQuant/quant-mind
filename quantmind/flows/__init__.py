"""Apex layer — composes configs / knowledge / preprocess on the SDK.

Semantic flows such as ``paper_flow`` return ``quantmind.knowledge`` values.
Deterministic operations such as ``collect_news`` return source-faithful
``quantmind.preprocess`` values. Cross-flow utilities live alongside:

- ``PaperFlow`` is the config-bound paper flow: ``PaperFlow(cfg)`` binds an
  immutable build config once and ``build(input)`` applies it per input,
  dispatching on the cfg **type** (``PaperStructureCfg`` → a self-contained
  ``PaperStructureTree``). ``batch_run(flow.build, inputs)`` runs every input
  under one unified setting.
- ``paper_flow`` is a thin compatibility function for the semantic
  chunk/summary shape (``PaperFlowResult``).
- ``batch_run`` runs any flow over a list of inputs with bounded
  concurrency and aggregated results, reporting aggregate token usage
  (and optionally priced cost) and enforcing the ``cfg`` budget guardrails.
- ``BatchResult`` is the shape returned by ``batch_run``; ``UsageSummary``
  and ``PriceRate`` describe its usage/cost fields, and
  ``BudgetExceededError`` marks inputs skipped after a budget tripped.
- ``UnsupportedContentTypeError`` is raised when a paper pipeline does not
  resolve a page-aware PDF.
- ``PaperStructureError`` is raised when structure building exceeds its
  runtime boundary.
"""

from quantmind.flows._usage import PriceRate, UsageSummary
from quantmind.flows.batch import BatchResult, BudgetExceededError, batch_run
from quantmind.flows.news import collect_news
from quantmind.flows.paper import (
    PaperFlow,
    PaperStructureError,
    UnsupportedContentTypeError,
    paper_flow,
)
from quantmind.knowledge import PaperCitationValidationError

__all__ = [
    "BatchResult",
    "BudgetExceededError",
    "PaperCitationValidationError",
    "PaperFlow",
    "PaperStructureError",
    "PriceRate",
    "UnsupportedContentTypeError",
    "UsageSummary",
    "batch_run",
    "collect_news",
    "paper_flow",
]
