"""Unified configuration management for QuantMind.

Simple, type-safe configuration system with YAML loading and environment variable substitution.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field

from quantmind.config.flows import (
    AnalyzerFlowConfig,
    BaseFlowConfig,
    QAFlowConfig,
    SummaryFlowConfig,
)
from quantmind.config.llm import LLMConfig
from quantmind.config.parsers import LlamaParserConfig, PDFParserConfig
from quantmind.config.sources import (
    ArxivSourceConfig,
    NewsSourceConfig,
    WebSourceConfig,
)
from quantmind.config.storage import LocalStorageConfig
from quantmind.config.taggers import LLMTaggerConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)

from dotenv import load_dotenv


class Setting(BaseModel):
    """Unified configuration for QuantMind - single instance pattern."""

    # Component configurations - single instances, not dictionaries
    source: Optional[
        Union[ArxivSourceConfig, NewsSourceConfig, WebSourceConfig]
    ] = None
    parser: Optional[Union[PDFParserConfig, LlamaParserConfig]] = None
    tagger: Optional[LLMTaggerConfig] = None
    storage: LocalStorageConfig = Field(default_factory=LocalStorageConfig)
    flow: Optional[
        Union[
            QAFlowConfig, SummaryFlowConfig, AnalyzerFlowConfig, BaseFlowConfig
        ]
    ] = None

    # Core configuration
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Global settings
    log_level: str = Field(
        default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )

    class Config:
        """Pydantic model configuration."""

        validate_assignment = True
        extra = "forbid"

    @classmethod
    def load_dotenv(cls, dotenv_path: Optional[str] = None) -> bool:
        """Load environment variables from .env file.

        Args:
            dotenv_path: Path to .env file. If None, auto-discovers .env file.

        Returns:
            True if .env file was found and loaded, False otherwise
        """
        if dotenv_path:
            # Load specific file
            env_path = Path(dotenv_path)
            if env_path.exists():
                load_dotenv(env_path)
                logger.info(f"Loaded environment from {env_path}")
                return True
            else:
                logger.warning(f"Dotenv file not found: {env_path}")
                return False
        else:
            # Auto-discover .env file
            current_dir = Path.cwd()
            env_paths = [
                current_dir / ".env",
                current_dir.parent / ".env",
            ]

            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(env_path)
                    logger.info(f"Loaded environment from {env_path}")
                    return True

            logger.debug("No .env file found")
            return False

    @classmethod
    def substitute_env_vars(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values.

        Supports syntax: ${ENV_VAR} or ${ENV_VAR:default_value}
        """

        def substitute_value(value: Any) -> Any:
            if isinstance(value, str):
                # Pattern: ${VAR} or ${VAR:default}
                pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

                def replacer(match):
                    env_var = match.group(1)
                    default_val = (
                        match.group(2) if match.group(2) is not None else ""
                    )
                    return os.getenv(env_var, default_val)

                return re.sub(pattern, replacer, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value

        return substitute_value(config_dict)

    @classmethod
    def from_yaml(
        cls, config_path: Union[str, Path], env_file: Optional[str] = None
    ) -> "Setting":
        """Load configuration from YAML file with environment variable substitution.

        Args:
            config_path: Path to YAML configuration file
            env_file: Optional path to .env file

        Returns:
            Configured Setting instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        # Load .env file first
        cls.load_dotenv(env_file)

        try:
            # Load YAML
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)

            if not isinstance(config_dict, dict):
                raise ValueError("Configuration file must contain a dictionary")

            # Substitute environment variables
            config_dict = cls.substitute_env_vars(config_dict)

            # Parse configuration
            return cls._parse_config(config_dict)

        except Exception as e:
            logger.error(
                f"Failed to load configuration from {config_path}: {e}"
            )
            raise

    @classmethod
    def _parse_config(cls, config_dict: Dict[str, Any]) -> "Setting":
        """Parse configuration dictionary into Setting instance."""
        # Configuration type registry
        CONFIG_REGISTRY = {
            "source": {
                "arxiv": ArxivSourceConfig,
                "news": NewsSourceConfig,
                "web": WebSourceConfig,
            },
            "parser": {
                "pdf": PDFParserConfig,
                "llama": LlamaParserConfig,
            },
            "tagger": {
                "llm": LLMTaggerConfig,
            },
            "storage": {
                "local": LocalStorageConfig,
            },
            "flow": {
                "qa": QAFlowConfig,
                "summary": SummaryFlowConfig,
                "analyzer": AnalyzerFlowConfig,
                "base": BaseFlowConfig,
            },
        }

        parsed = {}

        # Parse component configurations
        for component_name, type_registry in CONFIG_REGISTRY.items():
            if component_name in config_dict:
                component_data = config_dict[component_name]
                if isinstance(component_data, dict):
                    component_type = component_data.get("type")
                    component_config = component_data.get("config", {})

                    if component_type in type_registry:
                        config_class = type_registry[component_type]
                        parsed[component_name] = config_class(
                            **component_config
                        )
                    else:
                        logger.warning(
                            f"Unknown {component_name} type: {component_type}"
                        )

        # Parse other configurations
        if "llm" in config_dict:
            parsed["llm"] = LLMConfig(**config_dict["llm"])

        # Copy simple fields
        if "log_level" in config_dict:
            parsed["log_level"] = config_dict["log_level"]

        return cls(**parsed)

    @classmethod
    def create_default(cls) -> "Setting":
        """Create default configuration with sensible defaults."""
        return cls(
            source=ArxivSourceConfig(
                max_results=100,
                sort_by="submittedDate",
                sort_order="descending",
            ),
            parser=PDFParserConfig(
                method="pymupdf",
                download_pdfs=True,
                extract_tables=True,
            ),
            storage=LocalStorageConfig(),
        )

    def save_to_yaml(self, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file.

        Args:
            config_path: Path to save configuration to
        """
        config_path = Path(config_path)
        config_dict = self._export_config()

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            logger.info(f"Saved configuration to {config_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration to {config_path}: {e}")
            raise

    def _export_config(self) -> Dict[str, Any]:
        """Export configuration to dictionary format suitable for YAML."""

        def serialize_value(value: Any) -> Any:
            """Recursively serialize values."""
            if isinstance(value, Path):
                return str(value)
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            else:
                return value

        def serialize_component(component, component_type_map):
            if component is None:
                return None

            # Find component type
            component_class = type(component)
            component_type = None
            for type_name, type_class in component_type_map.items():
                if component_class == type_class:
                    component_type = type_name
                    break

            if component_type is None:
                return None

            # Serialize config, excluding sensitive fields
            config_dict = component.model_dump(exclude_none=True)
            config_dict.pop("api_key", None)  # Remove sensitive data

            # Convert Path objects to strings
            config_dict = serialize_value(config_dict)

            return {"type": component_type, "config": config_dict}

        # Type mappings for export
        type_maps = {
            "source": {
                "arxiv": ArxivSourceConfig,
                "news": NewsSourceConfig,
                "web": WebSourceConfig,
            },
            "parser": {"pdf": PDFParserConfig, "llama": LlamaParserConfig},
            "tagger": {"llm": LLMTaggerConfig},
            "storage": {"local": LocalStorageConfig},
            "flow": {
                "qa": QAFlowConfig,
                "summary": SummaryFlowConfig,
                "analyzer": AnalyzerFlowConfig,
                "base": BaseFlowConfig,
            },
        }

        config_dict = {}

        # Export components
        for component_name, type_map in type_maps.items():
            component = getattr(self, component_name, None)
            serialized = serialize_component(component, type_map)
            if serialized:
                config_dict[component_name] = serialized

        # Export LLM config (exclude sensitive data)
        config_dict["llm"] = self.llm.model_dump(exclude={"api_key"})

        # Export simple fields
        config_dict["log_level"] = self.log_level

        return config_dict


# Factory functions for convenience
def load_config(
    config_path: Union[str, Path], env_file: Optional[str] = None
) -> Setting:
    """Load configuration from YAML file."""
    return Setting.from_yaml(config_path, env_file)


def create_default_config() -> Setting:
    """Create default configuration."""
    return Setting.create_default()


# Export public API
__all__ = [
    "Setting",
    "load_config",
    "create_default_config",
]
