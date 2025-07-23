"""Configuration management for QuantMind."""

from .flows import (
    AnalyzerFlowConfig,
    BaseFlowConfig,
    QAFlowConfig,
    SummaryFlowConfig,
)
from .llm import LLMConfig
from .parsers import LlamaParserConfig, PDFParserConfig
from .settings import (
    Setting,
    create_default_config,
    load_config,
)
from .sources import (
    ArxivSourceConfig,
    BaseSourceConfig,
    NewsSourceConfig,
    WebSourceConfig,
)
from .storage import BaseStorageConfig, LocalStorageConfig
from .taggers import LLMTaggerConfig

__all__ = [
    # Core Settings
    "Setting",
    # LLM Configuration
    "LLMConfig",
    # Tagger Configurations
    "LLMTaggerConfig",
    # Parser Configurations
    "PDFParserConfig",
    "LlamaParserConfig",
    # Source Configurations
    "BaseSourceConfig",
    "ArxivSourceConfig",
    "NewsSourceConfig",
    "WebSourceConfig",
    # Storage Configurations
    "BaseStorageConfig",
    "LocalStorageConfig",
    # Flow Configurations
    "BaseFlowConfig",
    "QAFlowConfig",
    "SummaryFlowConfig",
    "AnalyzerFlowConfig",
    # Utility Functions
    "create_default_config",
    "load_config",
]
