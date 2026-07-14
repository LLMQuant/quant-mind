"""Apex layer — composes configs / knowledge / preprocess on the SDK.

Each flow function (``paper_flow``, ``news_flow``, future ``earnings_flow``)
takes a typed input and a ``<Name>FlowCfg`` and returns its domain result.
Cross-flow utilities live alongside:

- ``batch_run`` runs any flow over a list of inputs with bounded
  concurrency and aggregated results.
- ``BatchResult`` is the shape returned by ``batch_run``.
- ``UnsupportedContentTypeError`` is raised when ``paper_flow`` cannot
  route fetched bytes through the format layer.
"""

from quantmind.flows.batch import BatchResult, batch_run
from quantmind.flows.news import (
    NewsArtifact,
    NewsBatch,
    NewsDocument,
    NewsFailure,
    news_flow,
)
from quantmind.flows.paper import UnsupportedContentTypeError, paper_flow

__all__ = [
    "BatchResult",
    "NewsArtifact",
    "NewsBatch",
    "NewsDocument",
    "NewsFailure",
    "UnsupportedContentTypeError",
    "batch_run",
    "news_flow",
    "paper_flow",
]
