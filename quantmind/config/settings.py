"""Configuration management for QuantMind."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

from quantmind.config.parsers import (
    BaseParserConfig,
    LlamaParserConfig,
    PDFParserConfig,
)
from quantmind.utils.env import EnvConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SourceConfig:
    """Configuration for content sources."""

    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ComponentConfig:
    """Base configuration for a component."""

    name: str
    type: str
    config: Union[BaseParserConfig, Dict[str, Any]] = field(
        default_factory=dict
    )
    enabled: bool = True


@dataclass
class TaggerConfig:
    """Configuration for content taggers."""

    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class StorageConfig:
    """Configuration for storage backends."""

    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""

    max_workers: int = 4
    retry_attempts: int = 3
    timeout: int = 300
    enable_deduplication: bool = True
    quality_threshold: float = 0.5


@dataclass
class Settings:
    """Main configuration settings for QuantMind."""

    # Component configurations
    sources: Dict[str, SourceConfig] = field(default_factory=dict)
    parsers: Dict[str, ComponentConfig] = field(default_factory=dict)
    taggers: Dict[str, TaggerConfig] = field(default_factory=dict)
    storages: Dict[str, StorageConfig] = field(default_factory=dict)

    # Workflow configuration
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)

    # Global settings
    log_level: str = "INFO"
    data_dir: str = "./data"
    temp_dir: str = "/tmp"

    # API configurations
    openai_api_key: Optional[str] = None
    llama_cloud_api_key: Optional[str] = None
    arxiv_max_results: int = 100

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Settings":
        """Create Settings from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Settings instance
        """
        settings = cls()

        # Load sources
        for name, source_data in config_dict.get("sources", {}).items():
            settings.sources[name] = SourceConfig(
                name=name,
                type=source_data.get("type", "unknown"),
                config=source_data.get("config", {}),
                enabled=source_data.get("enabled", True),
            )

        # Load parsers
        for name, parser_data in config_dict.get("parsers", {}).items():
            parser_type = parser_data.get("type", "unknown")
            parser_config_data = parser_data.get("config", {})

            # Create appropriate Pydantic config based on parser type
            if parser_type == "LlamaParser":
                parser_config = LlamaParserConfig(**parser_config_data)
            elif parser_type == "PDFParser":
                parser_config = PDFParserConfig(**parser_config_data)
            else:
                # For unknown types, keep as dict for backward compatibility
                parser_config = parser_config_data

            settings.parsers[name] = ComponentConfig(
                name=name,
                type=parser_type,
                config=parser_config,
                enabled=parser_data.get("enabled", True),
            )

        # Load taggers
        for name, tagger_data in config_dict.get("taggers", {}).items():
            settings.taggers[name] = TaggerConfig(
                name=name,
                type=tagger_data.get("type", "unknown"),
                config=tagger_data.get("config", {}),
                enabled=tagger_data.get("enabled", True),
            )

        # Load storages
        for name, storage_data in config_dict.get("storages", {}).items():
            settings.storages[name] = StorageConfig(
                name=name,
                type=storage_data.get("type", "unknown"),
                config=storage_data.get("config", {}),
                enabled=storage_data.get("enabled", True),
            )

        # Load workflow configuration
        workflow_data = config_dict.get("workflow", {})
        settings.workflow = WorkflowConfig(
            max_workers=workflow_data.get("max_workers", 4),
            retry_attempts=workflow_data.get("retry_attempts", 3),
            timeout=workflow_data.get("timeout", 300),
            enable_deduplication=workflow_data.get(
                "enable_deduplication", True
            ),
            quality_threshold=workflow_data.get("quality_threshold", 0.5),
        )

        # Load global settings
        settings.log_level = config_dict.get("log_level", "INFO")
        settings.data_dir = config_dict.get("data_dir", "./data")
        settings.temp_dir = config_dict.get("temp_dir", "/tmp")
        settings.openai_api_key = config_dict.get("openai_api_key")
        settings.llama_cloud_api_key = config_dict.get("llama_cloud_api_key")
        settings.arxiv_max_results = config_dict.get("arxiv_max_results", 100)

        return settings

    def to_dict(self) -> Dict[str, Any]:
        """Convert Settings to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "sources": {
                name: {
                    "type": source.type,
                    "config": source.config,
                    "enabled": source.enabled,
                }
                for name, source in self.sources.items()
            },
            "parsers": {
                name: {
                    "type": parser.type,
                    "config": parser.config.model_dump()
                    if hasattr(parser.config, "model_dump")
                    else parser.config,
                    "enabled": parser.enabled,
                }
                for name, parser in self.parsers.items()
            },
            "taggers": {
                name: {
                    "type": tagger.type,
                    "config": tagger.config,
                    "enabled": tagger.enabled,
                }
                for name, tagger in self.taggers.items()
            },
            "storages": {
                name: {
                    "type": storage.type,
                    "config": storage.config,
                    "enabled": storage.enabled,
                }
                for name, storage in self.storages.items()
            },
            "workflow": {
                "max_workers": self.workflow.max_workers,
                "retry_attempts": self.workflow.retry_attempts,
                "timeout": self.workflow.timeout,
                "enable_deduplication": self.workflow.enable_deduplication,
                "quality_threshold": self.workflow.quality_threshold,
            },
            "log_level": self.log_level,
            "data_dir": self.data_dir,
            "temp_dir": self.temp_dir,
            "openai_api_key": self.openai_api_key,
            "llama_cloud_api_key": self.llama_cloud_api_key,
            "arxiv_max_results": self.arxiv_max_results,
        }

    def get_enabled_sources(self) -> Dict[str, SourceConfig]:
        """Get enabled source configurations.

        Returns:
            Dictionary of enabled sources
        """
        return {
            name: config
            for name, config in self.sources.items()
            if config.enabled
        }

    def get_enabled_parsers(self) -> Dict[str, ComponentConfig]:
        """Get enabled parser configurations.

        Returns:
            Dictionary of enabled parsers
        """
        return {
            name: config
            for name, config in self.parsers.items()
            if config.enabled
        }

    def get_enabled_taggers(self) -> Dict[str, TaggerConfig]:
        """Get enabled tagger configurations.

        Returns:
            Dictionary of enabled taggers
        """
        return {
            name: config
            for name, config in self.taggers.items()
            if config.enabled
        }

    def get_enabled_storages(self) -> Dict[str, StorageConfig]:
        """Get enabled storage configurations.

        Returns:
            Dictionary of enabled storages
        """
        return {
            name: config
            for name, config in self.storages.items()
            if config.enabled
        }


def load_config(config_path: Union[str, Path]) -> Settings:
    """Load configuration from a file.

    Args:
        config_path: Path to configuration file (YAML or JSON)

    Returns:
        Settings instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                config_dict = yaml.safe_load(f)
            elif config_path.suffix.lower() == ".json":
                import json

                config_dict = json.load(f)
            else:
                raise ValueError(
                    f"Unsupported config file format: {config_path.suffix}"
                )

        if not isinstance(config_dict, dict):
            raise ValueError("Configuration file must contain a dictionary")

        # Apply environment variable overrides
        config_dict = _apply_env_overrides(config_dict)

        settings = Settings.from_dict(config_dict)
        logger.info(f"Loaded configuration from {config_path}")

        return settings

    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.

    Args:
        config_dict: Base configuration dictionary

    Returns:
        Configuration with environment overrides applied
    """
    # Ensure environment is loaded (including .env files)
    EnvConfig.load_dotenv()

    # Define environment variable mappings
    env_mappings = {
        "QUANTMIND_LOG_LEVEL": "log_level",
        "QUANTMIND_DATA_DIR": "data_dir",
        "QUANTMIND_TEMP_DIR": "temp_dir",
        "OPENAI_API_KEY": "openai_api_key",
        "LLAMA_CLOUD_API_KEY": "llama_cloud_api_key",
        "QUANTMIND_ARXIV_MAX_RESULTS": "arxiv_max_results",
        "QUANTMIND_MAX_WORKERS": "workflow.max_workers",
        "QUANTMIND_RETRY_ATTEMPTS": "workflow.retry_attempts",
        "QUANTMIND_TIMEOUT": "workflow.timeout",
    }

    for env_var, config_key in env_mappings.items():
        env_value = EnvConfig.get_env_var(env_var)
        if env_value is not None:
            # Handle nested keys
            keys = config_key.split(".")
            current = config_dict

            # Navigate to parent
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set value with type conversion
            final_key = keys[-1]
            if env_var in [
                "QUANTMIND_ARXIV_MAX_RESULTS",
                "QUANTMIND_MAX_WORKERS",
                "QUANTMIND_RETRY_ATTEMPTS",
                "QUANTMIND_TIMEOUT",
            ]:
                current[final_key] = int(env_value)
            else:
                current[final_key] = env_value

    return config_dict


def create_default_config() -> Settings:
    """Create default configuration.

    Returns:
        Default Settings instance
    """
    settings = Settings()

    # Default source configuration
    settings.sources["arxiv"] = SourceConfig(
        name="arxiv",
        type="ArxivSource",
        config={"max_results": 100, "sort_by": "SubmittedDate"},
    )

    # Default parser configurations
    settings.parsers["pdf"] = ComponentConfig(
        name="pdf",
        type="PDFParser",
        config=PDFParserConfig(
            method="pymupdf",
            download_pdfs=True,
            max_file_size_mb=50,
        ),
    )

    settings.parsers["llama"] = ComponentConfig(
        name="llama",
        type="LlamaParser",
        config=LlamaParserConfig(
            result_type="markdown",
            parsing_mode="fast",
            max_file_size_mb=50,
        ),
        enabled=False,  # Disabled by default (requires API key)
    )

    # Default tagger configurations
    settings.taggers["rule"] = TaggerConfig(
        name="rule", type="RuleTagger", config={"case_sensitive": False}
    )

    settings.taggers["llm"] = TaggerConfig(
        name="llm",
        type="LLMTagger",
        config={
            "model_type": "openai",
            "model_name": "gpt-4",
            "temperature": 0.0,
        },
        enabled=False,  # Disabled by default (requires API key)
    )

    # Default storage configuration
    settings.storages["json"] = StorageConfig(
        name="json",
        type="JSONStorage",
        config={
            "storage_dir": "./data",
            "auto_backup": True,
            "max_backup_count": 5,
        },
    )

    return settings


def save_config(settings: Settings, config_path: Union[str, Path]) -> None:
    """Save configuration to a file.

    Args:
        settings: Settings instance to save
        config_path: Path to save configuration to
    """
    config_path = Path(config_path)
    config_dict = settings.to_dict()

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.suffix.lower() in [".yml", ".yaml"]:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            else:
                import json

                json.dump(config_dict, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved configuration to {config_path}")

    except Exception as e:
        logger.error(f"Failed to save configuration to {config_path}: {e}")
        raise
