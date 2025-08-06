"""Flow configuration registry for dynamic type resolution."""

import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Type

from quantmind.config.flows import BaseFlowConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


# TODO: Add the corresponding unittests.
class FlowConfigRegistry:
    """Registry for flow configuration classes enabling dynamic loading."""

    _instance: Optional["FlowConfigRegistry"] = None
    _registry: Dict[str, Type[BaseFlowConfig]] = {}

    def __new__(cls) -> "FlowConfigRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize registry with built-in flow types."""
        if not hasattr(self, "_initialized"):
            self._register_builtin_flows()
            self._initialized = True

    def _register_builtin_flows(self):
        """Register built-in flow configuration types."""
        from quantmind.config.flows import BaseFlowConfig, SummaryFlowConfig

        self._registry["base"] = BaseFlowConfig
        self._registry["summary"] = SummaryFlowConfig

        logger.debug("Registered built-in flow types: base, summary")

    def register(
        self, flow_type: str, config_class: Type[BaseFlowConfig]
    ) -> None:
        """Register a flow configuration class.

        Args:
            flow_type: String identifier for the flow type
            config_class: Configuration class (must inherit from BaseFlowConfig)

        Raises:
            ValueError: If config_class doesn't inherit from BaseFlowConfig
        """
        if not issubclass(config_class, BaseFlowConfig):
            raise ValueError(
                f"Flow config class {config_class.__name__} must inherit from BaseFlowConfig"
            )

        self._registry[flow_type] = config_class
        logger.debug(
            f"Registered flow type '{flow_type}' -> {config_class.__name__}"
        )

    def get_config_class(self, flow_type: str) -> Type[BaseFlowConfig]:
        """Get configuration class for a flow type.

        Args:
            flow_type: String identifier for the flow type

        Returns:
            Configuration class

        Raises:
            KeyError: If flow type is not registered
        """
        if flow_type not in self._registry:
            raise KeyError(f"Unknown flow type: {flow_type}")
        return self._registry[flow_type]

    def list_types(self) -> Dict[str, str]:
        """List all registered flow types.

        Returns:
            Dictionary mapping flow type to class name
        """
        return {
            flow_type: config_class.__name__
            for flow_type, config_class in self._registry.items()
        }

    def auto_discover_flows(self, search_paths: list[Path]) -> None:
        """Auto-discover and register flow configurations from specified paths.

        Args:
            search_paths: List of directories to search for flow configurations
        """
        for search_path in search_paths:
            if not search_path.exists():
                logger.debug(f"Search path does not exist: {search_path}")
                continue

            self._discover_flows_in_path(search_path)

    def _discover_flows_in_path(self, path: Path) -> None:
        """Discover flows in a specific path."""
        # Look for flow.py files in subdirectories
        for flow_file in path.rglob("flow.py"):
            try:
                self._load_flow_from_file(flow_file)
            except Exception as e:
                logger.warning(f"Failed to load flow from {flow_file}: {e}")

    def _load_flow_from_file(self, flow_file: Path) -> None:
        """Load flow configuration from a Python file."""
        # Convert path to module name
        relative_path = flow_file.relative_to(Path.cwd())
        module_path_parts = list(relative_path.parts[:-1]) + [
            relative_path.stem
        ]
        module_name = ".".join(module_path_parts)

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Look for classes that inherit from BaseFlowConfig
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, BaseFlowConfig)
                    and obj != BaseFlowConfig
                    and obj.__module__ == module_name
                ):
                    # Infer flow type from class name (e.g., GreetingFlowConfig -> greeting)
                    flow_type = self._infer_flow_type(name)
                    self.register(flow_type, obj)

        except ImportError as e:
            logger.warning(f"Could not import module {module_name}: {e}")

    def _infer_flow_type(self, class_name: str) -> str:
        """Infer flow type from class name.

        Args:
            class_name: Class name (e.g., "GreetingFlowConfig")

        Returns:
            Flow type string (e.g., "greeting")
        """
        # Remove "FlowConfig" suffix and convert to lowercase
        if class_name.endswith("FlowConfig"):
            flow_type = class_name[:-10]  # Remove "FlowConfig"
        elif class_name.endswith("Config"):
            flow_type = class_name[:-6]  # Remove "Config"
        else:
            flow_type = class_name

        # Convert from CamelCase to snake_case
        import re

        flow_type = re.sub("([A-Z]+)", r"_\1", flow_type).lower().strip("_")

        return flow_type


# Global registry instance
flow_registry = FlowConfigRegistry()


def register_flow_config(flow_type: str):
    """Decorator to register a flow configuration class.

    Args:
        flow_type: String identifier for the flow type

    Example:
        @register_flow_config("greeting")
        class GreetingFlowConfig(BaseFlowConfig):
            pass
    """

    def decorator(config_class: Type[BaseFlowConfig]):
        flow_registry.register(flow_type, config_class)
        return config_class

    return decorator
