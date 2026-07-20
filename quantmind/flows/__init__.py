"""Apex layer — composes configs / knowledge / preprocess on the SDK.

Semantic flows such as ``paper_flow`` return ``quantmind.knowledge`` values.
Deterministic operations such as ``collect_news`` return source-faithful
``quantmind.preprocess`` values. Cross-flow utilities live alongside:

- ``PaperFlow`` is the document-scoped handle grouping the finished paper
  pipelines (``open`` fetch-and-parses once; ``build_structure`` and
  ``extract_knowledge`` are pure pipelines over that immutable source).
- ``paper_flow`` is a thin compatibility function delegating to
  ``PaperFlow.open(...).extract_knowledge(...)``.
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
    paper_flow,
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
    "paper_flow",
]
