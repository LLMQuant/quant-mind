import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import dotenv
from llama_cloud_services import LlamaParse
from llama_cloud_services.parse.types import JobResult

from .parse_tool import STRUCTURED_TYPES, ParseTool

dotenv.load_dotenv()


class ParsingMode(Enum):
    """The mode of parsing to use."""

    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"


class ResultType(str, Enum):
    """The result type for the parser."""

    TXT = "text"
    MD = "markdown"
    JSON = "json"
    STRUCTURED = "structured"


class LlamaParser(ParseTool):
    """LlamaParser is a wrapper around the LlamaParse API.

    For more information, see: https://docs.cloud.llamaindex.ai/llamaparse/getting_started
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: ResultType = ResultType.MD,
        parsing_mode: ParsingMode = ParsingMode.FAST,
        system_prompt: Optional[str] = None,
        system_prompt_append: Optional[str] = None,
        **kwargs,
    ):
        self.api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
        if not self.api_key:
            raise ValueError("LLAMA_CLOUD_API_KEY is not set")

        self.parse_config = kwargs
        self.parsing_mode = parsing_mode
        self.result_type = result_type
        self.system_prompt = system_prompt
        self.system_prompt_append = system_prompt_append
        self._build_config()

        self.llama_parse = LlamaParse(api_key=self.api_key, **self.parse_config)

    def _build_config(self):
        """Build the LlamaParse config based on parameters."""
        self.parse_config["result_type"] = self.result_type

        if self.parsing_mode == ParsingMode.FAST:
            self.parse_config["fast_mode"] = True
            self.parse_config["premium_mode"] = False
        elif self.parsing_mode == ParsingMode.BALANCED:
            self.parse_config["fast_mode"] = False
            self.parse_config["premium_mode"] = False
        elif self.parsing_mode == ParsingMode.PREMIUM:
            self.parse_config["fast_mode"] = False
            self.parse_config["premium_mode"] = True

        if self.system_prompt:
            self.parse_config["system_prompt"] = self.system_prompt
        if self.system_prompt_append:
            self.parse_config["system_prompt_append"] = (
                self.system_prompt_append
            )

    def parse(
        self, source: Union[str, Path, bytes], **kwargs: Any
    ) -> JobResult:
        """Parse the source into a JobResult object."""
        return self.llama_parse.parse(source, **kwargs)

    def get_format(self) -> STRUCTURED_TYPES:
        return STRUCTURED_TYPES.LLAMA_PARSE


if __name__ == "__main__":
    parser = LlamaParser()
    result = parser.parse(Path("test-pdf.pdf"))
    print(result)
