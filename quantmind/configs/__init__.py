"""quantmind.configs — public operation configuration + input types.

Public operations use a ``BaseFlowCfg`` subclass and a typed input model or
discriminated union. All cfg / input classes live here so that:

  - YAML / CLI users see a single import surface,
  - JSON schemas can be exported uniformly (for IDE autocomplete),
  - the magic-input resolver (PR5) has one introspection target.
"""

from quantmind.configs.base import BaseFlowCfg, BaseInput
from quantmind.configs.earnings import EarningsFlowCfg, EarningsInput
from quantmind.configs.news import NewsCollectionCfg, NewsWindow
from quantmind.configs.paper import PaperFlowCfg, PaperInput
from quantmind.configs.retrieval import AgenticRetrievalCfg, RetrievalCfg
from quantmind.configs.structure import PaperStructureCfg

__all__ = [
    "AgenticRetrievalCfg",
    "BaseFlowCfg",
    "BaseInput",
    "EarningsFlowCfg",
    "EarningsInput",
    "NewsCollectionCfg",
    "NewsWindow",
    "PaperFlowCfg",
    "PaperInput",
    "PaperStructureCfg",
    "RetrievalCfg",
]
