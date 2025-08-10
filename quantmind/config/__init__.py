"""Configuration management for QuantMind."""

from .flows import (
    BaseFlowConfig,
    SummaryFlowConfig,
)
from .llm import LLMConfig
from .embedding import EmbeddingConfig
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
    "EmbeddingConfig",
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
    "SummaryFlowConfig",
    # Utility Functions
    "create_default_config",
    "load_config",
]
