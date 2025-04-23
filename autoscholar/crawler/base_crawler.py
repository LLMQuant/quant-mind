from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import yaml

from autoscholar.utils.logger import setup_logger

# Set up logger
logger = setup_logger(__name__)


class BaseCrawler(ABC):
    """Base class for implementing crawlers for different sources.

    This abstract class provides a minimal interface for crawlers.
    Subclasses can implement their own specific functionality and workflow.
    """

    def __init__(self, **kwargs):
        """Initialize the crawler with optional parameters.

        Parameters:
        ----------
        **kwargs : Any
            Optional parameters that can be used by subclasses.
        """
        self.config = kwargs
        self.output_dir = kwargs.get("output_dir", "data")

        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config_file(cls, config_path: Path, **kwargs) -> "BaseCrawler":
        """Create a crawler instance from a configuration file.

        Parameters:
        ----------
        config_path : Path
            Path to the configuration file
        **kwargs : Any
            Additional parameters to override config file settings

        Returns:
        -------
        BaseCrawler
            Configured crawler instance
        """
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Update config with any overrides
        config.update(kwargs)
        return cls(**config)

    @abstractmethod
    def run(self, **kwargs) -> None:
        """Execute the crawler workflow.

        This is the only required method that subclasses must implement.
        The implementation can be as simple or complex as needed.

        Parameters:
        ----------
        **kwargs : Any
            Optional parameters that can be used by subclasses.
        """
        pass
