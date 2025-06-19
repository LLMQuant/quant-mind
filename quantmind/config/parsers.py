"""Configuration models for parsers."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ParsingMode(str, Enum):
    """The mode of parsing to use."""

    FAST = "fast"
    BALANCED = "balanced"
    PREMIUM = "premium"


class ResultType(str, Enum):
    """The result type for the parser."""

    TXT = "text"
    MD = "markdown"
    JSON = "json"


class BaseParserConfig(BaseModel):
    """Base configuration for all parsers."""

    max_file_size_mb: int = Field(default=50, ge=1, le=100)
    timeout: int = Field(default=120, ge=10, le=600)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    enable_caching: bool = Field(default=True)

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        use_enum_values = True


class PDFParserConfig(BaseParserConfig):
    """Configuration for PDF parser."""

    method: str = Field(default="pymupdf")
    download_pdfs: bool = Field(default=True)
    extract_images: bool = Field(default=False)
    extract_tables: bool = Field(default=True)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate parsing method."""
        valid_methods = ["pymupdf", "pdfplumber", "marker"]
        if v not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}")
        return v


class LlamaParserConfig(BaseParserConfig):
    """Configuration for LlamaParser."""

    # API settings
    api_key: Optional[str] = Field(default=None)
    result_type: ResultType = Field(default=ResultType.MD)
    parsing_mode: ParsingMode = Field(default=ParsingMode.FAST)

    # Custom prompts
    system_prompt: Optional[str] = Field(default=None)
    system_prompt_append: Optional[str] = Field(default=None)

    # Performance settings
    num_workers: Optional[int] = Field(default=None, ge=1, le=10)
    verbose: bool = Field(default=False)
    language: Optional[str] = Field(default=None)

    # Page selection
    target_pages: Optional[List[int]] = Field(default=None)
    split_by_page: bool = Field(default=False)

    # Caching options
    invalidate_cache: bool = Field(default=False)
    do_not_cache: bool = Field(default=False)

    # Advanced settings
    check_interval: Optional[int] = Field(default=None, ge=1, le=60)
    max_timeout: Optional[int] = Field(default=None, ge=60, le=3600)
    auto_mode_trigger_on_text_length: Optional[int] = Field(
        default=None, ge=100, le=10000
    )

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        """Validate language code."""
        if v is None:
            return v

        # Common language codes
        valid_languages = [
            "en",
            "zh",
            "es",
            "fr",
            "de",
            "ja",
            "ko",
            "ru",
            "ar",
            "pt",
            "it",
        ]
        if v not in valid_languages:
            raise ValueError(
                f"language must be one of {valid_languages} or None"
            )
        return v

    @field_validator("target_pages")
    @classmethod
    def validate_target_pages(
        cls, v: Optional[List[int]]
    ) -> Optional[List[int]]:
        """Validate target pages."""
        if v is None:
            return v

        if not v:
            raise ValueError("target_pages cannot be empty if provided")

        for page in v:
            if page == 0:
                raise ValueError("page numbers start from 1, not 0")
            if page < -1:
                raise ValueError("negative page numbers must be -1 or greater")

        return v

    def get_llama_parse_config(self) -> Dict[str, Any]:
        """Get configuration for LlamaParse initialization.

        Returns:
            Dictionary with LlamaParse configuration parameters
        """
        # Handle enum values properly
        result_type_value = (
            self.result_type
            if isinstance(self.result_type, str)
            else self.result_type.value
        )
        parsing_mode_value = (
            self.parsing_mode
            if isinstance(self.parsing_mode, str)
            else self.parsing_mode.value
        )

        config = {
            "result_type": result_type_value,
        }

        # Set parsing mode
        if parsing_mode_value == "fast":
            config["fast_mode"] = True
            config["premium_mode"] = False
        elif parsing_mode_value == "balanced":
            config["fast_mode"] = False
            config["premium_mode"] = False
        elif parsing_mode_value == "premium":
            config["fast_mode"] = False
            config["premium_mode"] = True

        # Add system prompts if provided
        if self.system_prompt:
            config["system_prompt"] = self.system_prompt
        if self.system_prompt_append:
            config["system_prompt_append"] = self.system_prompt_append

        # Add optional parameters
        optional_fields = [
            "num_workers",
            "verbose",
            "language",
            "target_pages",
            "split_by_page",
            "invalidate_cache",
            "do_not_cache",
            "check_interval",
            "max_timeout",
            "auto_mode_trigger_on_text_length",
        ]

        for field in optional_fields:
            value = getattr(self, field)
            if value is not None:
                config[field] = value

        return config
