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
from quantmind.configs.paper import PaperInput, PaperSemanticCfg
from quantmind.configs.retrieval import RetrievalCfg
from quantmind.configs.structure import PaperStructureCfg

__all__ = [
    "BaseFlowCfg",
    "BaseInput",
    "EarningsFlowCfg",
    "EarningsInput",
    "NewsCollectionCfg",
    "NewsWindow",
    "PaperSemanticCfg",
    "PaperInput",
    "PaperStructureCfg",
    "RetrievalCfg",
]
