"""Apex layer — composes configs / knowledge / preprocess on the SDK.

Semantic flows such as ``paper_flow`` return ``quantmind.knowledge`` values.
Deterministic operations such as ``collect_news`` return source-faithful
``quantmind.preprocess`` values. Cross-flow utilities live alongside:

- ``batch_run`` runs any flow over a list of inputs with bounded
  concurrency and aggregated results.
- ``BatchResult`` is the shape returned by ``batch_run``.
- ``UnsupportedContentTypeError`` is raised when ``paper_flow`` does not
  resolve a page-aware PDF.
"""

from quantmind.flows.batch import BatchResult, batch_run
from quantmind.flows.news import collect_news
from quantmind.flows.paper import UnsupportedContentTypeError, paper_flow
from quantmind.knowledge import PaperCitationValidationError

__all__ = [
    "BatchResult",
    "PaperCitationValidationError",
    "UnsupportedContentTypeError",
    "batch_run",
    "collect_news",
    "paper_flow",
    "run_fallback_policy",
]
