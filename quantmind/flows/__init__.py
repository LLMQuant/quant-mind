"""Apex layer — composes configs / knowledge / preprocess on the SDK.

The config-bound ``PaperFlow`` returns ``quantmind.knowledge`` values.
Deterministic operations such as ``collect_news`` return source-faithful
``quantmind.preprocess`` values. Cross-flow utilities live alongside:

- ``PaperFlow`` is the config-bound paper flow: ``PaperFlow(cfg)`` binds an
  immutable build config once and ``build(input)`` applies it per input,
  dispatching on the cfg **type** (``PaperStructureCfg`` → a self-contained
  ``PaperStructureTree``; ``PaperSemanticCfg`` → a source-first
  ``PaperSemanticResult``). ``batch_run(flow.build, inputs)`` runs every input
  under one unified setting.
- ``batch_run`` runs any flow over a list of inputs with bounded
  concurrency and aggregated results.
- ``BatchResult`` is the shape returned by ``batch_run``.
- ``UnsupportedContentTypeError`` is raised when a paper pipeline does not
  resolve a page-aware PDF.
- ``PaperStructureError`` is raised when structure building exceeds its
  runtime boundary.
"""

from quantmind.flows.batch import BatchResult, batch_run
from quantmind.flows.news import collect_news
from quantmind.flows.paper import (
    PaperFlow,
    PaperStructureError,
    UnsupportedContentTypeError,
)
from quantmind.knowledge import PaperCitationValidationError

__all__ = [
    "BatchResult",
    "PaperCitationValidationError",
    "PaperFlow",
    "PaperStructureError",
    "UnsupportedContentTypeError",
    "batch_run",
    "collect_news",
]
