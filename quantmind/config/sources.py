"""Configuration models for sources."""

from pathlib import Path
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator
import arxiv


class BaseSourceConfig(BaseModel):
    """Base configuration for all sources."""

    max_results: int = Field(default=100, ge=1, le=1000)
    timeout: int = Field(default=30, ge=1, le=300)
    retry_attempts: int = Field(default=3, ge=0, le=10)
    proxies: Optional[dict] = Field(default=None)

    class Config:
        """Pydantic configuration."""

        validate_assignment = True


class ArxivSourceConfig(BaseSourceConfig):
    """Configuration for ArXiv source."""

    # API settings
    sort_by: str = Field(default="submittedDate")
    sort_order: str = Field(default="descending")

    # Download settings
    download_pdfs: bool = Field(default=False)
    download_dir: Optional[Path] = Field(default=None)

    # Rate limiting
    requests_per_second: float = Field(default=1.0, ge=0.1, le=10.0)

    # Content filtering
    include_categories: Optional[List[str]] = Field(default=None)
    exclude_categories: Optional[List[str]] = Field(default=None)
    min_abstract_length: int = Field(default=50, ge=0)

    # Language filtering
    languages: Optional[List[str]] = Field(default=None)

    @field_validator("sort_by")
    def validate_sort_by(cls, v):
        """Validate sort_by field."""
        valid_sorts = ["relevance", "lastUpdatedDate", "submittedDate"]
        if v not in valid_sorts:
            raise ValueError(f"sort_by must be one of {valid_sorts}")
        return v

    @field_validator("sort_order")
    def validate_sort_order(cls, v):
        """Validate sort_order field."""
        if v not in ["ascending", "descending"]:
            raise ValueError("sort_order must be 'ascending' or 'descending'")
        return v

    @field_validator("download_dir")
    def validate_download_dir(cls, v):
        """Validate and create download directory if needed."""
        if v is not None:
            v = Path(v)
            if not v.exists():
                v.mkdir(parents=True, exist_ok=True)
            elif not v.is_dir():
                raise ValueError(f"download_dir must be a directory: {v}")
        return v

    @field_validator("include_categories", "exclude_categories")
    def validate_categories(cls, v):
        """Validate arXiv categories."""
        if v is not None:
            # Common arXiv categories for validation
            valid_categories = [
                "cs.AI",
                "cs.CL",
                "cs.CV",
                "cs.LG",
                "cs.MA",
                "cs.NE",
                "stat.ML",
                "stat.AP",
                "stat.CO",
                "stat.ME",
                "stat.TH",
                "q-fin.CP",
                "q-fin.EC",
                "q-fin.GN",
                "q-fin.MF",
                "q-fin.PM",
                "q-fin.PR",
                "q-fin.RM",
                "q-fin.ST",
                "q-fin.TR",
                "math.PR",
                "math.ST",
                "math.OC",
                "math.NA",
                "econ.EM",
                "econ.GN",
                "econ.TH",
            ]
            for cat in v:
                if cat not in valid_categories:
                    # Allow but warn about unknown categories
                    pass
        return v

    def get_arxiv_sort_criterion(self) -> arxiv.SortCriterion:
        """Get arXiv sort criterion from config."""
        sort_map = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate,
        }
        return sort_map[self.sort_by]

    def get_arxiv_sort_order(self) -> arxiv.SortOrder:
        """Get arXiv sort order from config."""
        order_map = {
            "ascending": arxiv.SortOrder.Ascending,
            "descending": arxiv.SortOrder.Descending,
        }
        return order_map[self.sort_order]


class NewsSourceConfig(BaseSourceConfig):
    """Configuration for news sources."""

    # API settings
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)

    # Content filtering
    sources: Optional[List[str]] = Field(default=None)
    domains: Optional[List[str]] = Field(default=None)
    exclude_domains: Optional[List[str]] = Field(default=None)

    # Language and location
    language: str = Field(default="en")
    country: Optional[str] = Field(default=None)

    @field_validator("language")
    def validate_language(cls, v):
        """Validate language code."""
        # ISO 639-1 language codes
        valid_languages = [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "ru",
            "ja",
            "ko",
            "zh",
        ]
        if v not in valid_languages:
            raise ValueError(f"language must be one of {valid_languages}")
        return v


class WebSourceConfig(BaseSourceConfig):
    """Configuration for web scraping sources."""

    # Request settings
    user_agent: str = Field(default="QuantMind/1.0")
    headers: Optional[dict] = Field(default=None)
    cookies: Optional[dict] = Field(default=None)

    # Scraping settings
    follow_redirects: bool = Field(default=True)
    verify_ssl: bool = Field(default=True)

    # Content extraction
    selectors: Optional[dict] = Field(default=None)

    # Rate limiting
    delay_between_requests: float = Field(default=1.0, ge=0.0)

    @field_validator("delay_between_requests")
    def validate_delay(cls, v):
        """Ensure reasonable delay between requests."""
        if v < 0.1:
            raise ValueError(
                "delay_between_requests should be at least 0.1 seconds"
            )
        return v


# Source configuration registry
SOURCE_CONFIGS = {
    "arxiv": ArxivSourceConfig,
    "news": NewsSourceConfig,
    "web": WebSourceConfig,
}


def get_source_config(source_type: str) -> type:
    """Get configuration class for source type.

    Args:
        source_type: Type of source (e.g., 'arxiv', 'news')

    Returns:
        Configuration class for the source type

    Raises:
        ValueError: If source type is not supported
    """
    if source_type not in SOURCE_CONFIGS:
        raise ValueError(f"Unsupported source type: {source_type}")
    return SOURCE_CONFIGS[source_type]
