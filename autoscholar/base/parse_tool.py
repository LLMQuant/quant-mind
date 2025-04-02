from abc import ABC, abstractmethod
from typing import Dict, Union


class ParseTool(ABC):
    """Base abstract class for PDF parsing tools."""

    @abstractmethod
    def parse(self, file_path: str) -> Union[str, Dict]:
        """Parse PDF file into structured format.

        Args:
            file_path: Path to the PDF file

        Returns:
            Parsed content in the appropriate format (markdown string or JSON dict)
        """
        pass

    @abstractmethod
    def get_format(self) -> str:
        """Return the output format of this parser."""
        pass
