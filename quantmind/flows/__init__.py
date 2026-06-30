"""Apex layer — composes configs / knowledge / preprocess on the SDK.

Each flow function takes a typed input and a ``<Name>FlowCfg`` and returns a
knowledge item. The current production flow is finance-first (``paper_flow``),
but the apex-layer contract itself is reusable for future domain flows as long
as they follow the same typed ``(input, *, cfg, ...)`` pattern.

Cross-flow utilities live alongside:

- ``batch_run`` runs any flow over a list of inputs with bounded
  concurrency and aggregated results.
- ``BatchResult`` is the shape returned by ``batch_run``.
- ``UnsupportedContentTypeError`` is raised when ``paper_flow`` cannot
  route fetched bytes through the format layer.
"""

from quantmind.flows.batch import BatchResult, batch_run
from quantmind.flows.paper import UnsupportedContentTypeError, paper_flow

__all__ = [
    "BatchResult",
    "UnsupportedContentTypeError",
    "batch_run",
    "paper_flow",
]
