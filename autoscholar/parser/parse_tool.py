from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Union


class STRUCTURED_TYPES(Enum):
    """The types of structured output that can be returned by a parser."""

    MARKDOWN = "markdown"
    JSON = "json"
    LLAMA_PARSE = "llama_parse"


class ParseTool(ABC):
    """Base abstract class for parsing tools handling various inputs."""

    @abstractmethod
    def parse(
        self, source: Union[str, Path, bytes], **kwargs: Any
    ) -> Union[str, Dict, Any]:
        """Parse content from a given source into a structured format.

        Args:
            source: The source of the content. Can be a file path (str or Path)
                    or raw content (bytes). Subclasses should handle the
                    appropriate type(s) they support.
            **kwargs: Additional keyword arguments specific to the parser.

        Returns:
            Parsed content in the format specified by get_format().
        """
        pass

    @abstractmethod
    def get_format(self) -> STRUCTURED_TYPES:
        """Return the target output format of this parser."""
        pass
