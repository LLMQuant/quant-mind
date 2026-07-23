"""quantmind.configs — public operation configuration + input types.

Public operations use a ``BaseFlowCfg`` subclass and a typed input model or
discriminated union. All cfg / input classes live here so that:

  - YAML / CLI users see a single import surface,
  - JSON schemas can be exported uniformly (for IDE autocomplete),
  - the magic-input resolver (PR5) has one introspection target.
"""

from quantmind.configs.base import (
    ATLASCLOUD_BASE_URL,
    ATLASCLOUD_DEFAULT_CHAT_MODEL,
    ATLASCLOUD_DEFAULT_REASONING_MODEL,
    BaseFlowCfg,
    BaseInput,
    atlascloud_model,
    is_atlascloud_model,
    resolve_agent_model,
)
from quantmind.configs.earnings import EarningsFlowCfg, EarningsInput
from quantmind.configs.news import NewsCollectionCfg, NewsWindow
from quantmind.configs.paper import PaperFlowCfg, PaperInput
from quantmind.configs.retrieval import RetrievalCfg
from quantmind.configs.structure import PaperStructureCfg

__all__ = [
    "BaseFlowCfg",
    "BaseInput",
    "ATLASCLOUD_BASE_URL",
    "ATLASCLOUD_DEFAULT_CHAT_MODEL",
    "ATLASCLOUD_DEFAULT_REASONING_MODEL",
    "EarningsFlowCfg",
    "EarningsInput",
    "NewsCollectionCfg",
    "NewsWindow",
    "PaperFlowCfg",
    "PaperInput",
    "PaperStructureCfg",
    "RetrievalCfg",
    "atlascloud_model",
    "is_atlascloud_model",
    "resolve_agent_model",
]
