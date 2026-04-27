"""quantmind.configs — flow configuration + input types.

Each flow has a `<Name>FlowCfg` (extends `BaseFlowCfg`) and a `<Name>Input`
discriminated-union type. All cfg / input classes live here so that:

  - YAML / CLI users see a single import surface,
  - JSON schemas can be exported uniformly (for IDE autocomplete),
  - the magic-input resolver (PR5) has one introspection target.
"""

from quantmind.configs.base import BaseFlowCfg, BaseInput
from quantmind.configs.earnings import EarningsFlowCfg, EarningsInput
from quantmind.configs.news import NewsFlowCfg, NewsInput
from quantmind.configs.paper import PaperFlowCfg, PaperInput

__all__ = [
    "BaseFlowCfg",
    "BaseInput",
    "EarningsFlowCfg",
    "EarningsInput",
    "NewsFlowCfg",
    "NewsInput",
    "PaperFlowCfg",
    "PaperInput",
]
