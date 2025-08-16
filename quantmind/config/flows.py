"""Base flow configuration for QuantMind framework."""

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import yaml
from pydantic import BaseModel, Field

from quantmind.config.llm import LLMConfig
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


# ===== Base Flow Configuration =====
class BaseFlowConfig(BaseModel):
    """Base configuration for flows - only defines resources needed.

    This simplified config focuses on providing resources (LLM blocks and prompt templates)
    rather than orchestrating flow logic, which is now handled in code.
    """

    name: str
    llm_blocks: Dict[str, LLMConfig] = Field(default_factory=dict)
    prompt_templates: Dict[str, str] = Field(default_factory=dict)
    prompt_templates_path: Union[str, Path, None] = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize configuration after dataclass creation."""
        if self.prompt_templates_path:
            self._load_prompt_templates()

    def _load_prompt_templates(self):
        """Load prompt templates from YAML file."""
        logger.info(
            f"Loading prompt templates from {self.prompt_templates_path}"
        )

        path = Path(self.prompt_templates_path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt templates file not found: {path}")

        if path.suffix.lower() not in [".yaml", ".yml"]:
            raise ValueError(
                f"Prompt templates file must be a YAML file, got: {path.suffix}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Replace current prompt_templates with loaded ones
        templates = data.get("templates", {})
        if not templates:
            raise ValueError(f"No 'templates' section found in {path}")

        self.prompt_templates = templates


# ===== Summary Flow Configuration =====
class ChunkingStrategy(Enum):
    """Strategy for chunking content.

    Attributes:
        BY_SIZE: Chunk by size
        BY_SECTION: Chunk by section
    """

    BY_SIZE = "size"
    BY_SECTION = "section"
    BY_CUSTOM = "custom"


class SummaryFlowConfig(BaseFlowConfig):
    """Configuration for content summary generation flow."""

    use_chunking: bool = True
    chunk_size: int = 2000
    chunk_strategy: ChunkingStrategy = ChunkingStrategy.BY_SIZE
    chunk_custom_strategy: Union[Callable[[str], List[str]], None] = None

    def model_post_init(self, __context: Any) -> None:
        """Initialize default LLM blocks and templates for summary flow."""
        # First load prompt templates from path if specified
        super().model_post_init(__context)

        # Allow BY_SIZE and BY_CUSTOM strategies
        if self.chunk_strategy not in [
            ChunkingStrategy.BY_SIZE,
            ChunkingStrategy.BY_CUSTOM,
        ]:
            raise NotImplementedError(
                f"Chunking strategy {self.chunk_strategy} is not implemented for this flow."
            )

        if not self.llm_blocks:
            # Default LLM blocks for the two-stage summary process
            self.llm_blocks = {
                "cheap_summarizer": LLMConfig(
                    model="gpt-4o-mini", temperature=0.3, max_tokens=1000
                ),
                "powerful_combiner": LLMConfig(
                    model="gpt-4o", temperature=0.3, max_tokens=2000
                ),
            }

        if not self.prompt_templates:
            # Default prompt templates
            self.prompt_templates = {
                "summarize_chunk_template": (
                    "You are a financial research expert. Summarize the following content chunk "
                    "focusing on key insights, methodology, and findings. Keep it concise but comprehensive.\n\n"
                    "Content:\n{{ chunk_text }}\n\n"
                    "Summary:"
                ),
                "combine_summaries_template": (
                    "You are a financial research expert. Combine the following chunk summaries "
                    "into a coherent, comprehensive final summary. Eliminate redundancy and "
                    "create a well-structured overview.\n\n"
                    "Chunk Summaries:\n{{ summaries }}\n\n"
                    "Final Summary:"
                ),
            }


class PodcastFlowConfig(BaseFlowConfig):
    """Configuration for podcast generation flow."""

    num_speakers: int = 2
    speaker_languages: str = "en-us"
    summary_hint: str
