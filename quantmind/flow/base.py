"""Base flow abstract class for QuantMind framework."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from quantmind.config.flows import BaseFlowConfig
from quantmind.llm import create_llm_block
from quantmind.models.content import KnowledgeItem
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class BaseFlow(ABC):
    """Abstract base class for all LLM-based flows.

    All flows follow the pattern:
    1. build_prompt() - Create prompt from content and configuration using template system
    2. execute() - Process content using LLM and return structured results

    Key Features:
    - Template-based prompt engineering with {{variable}} syntax
    - Flexible system prompt configuration
    - Support for custom prompt building functions
    - Generic KnowledgeItem input (not just Paper)
    - Type-safe configuration with Pydantic validation

    The key difference between BaseFlow and LLMBlock:
    - LLMBlock: Provides basic LLM capabilities (generate_text, generate_structured_output)
    - BaseFlow: Implements business logic and flows using LLMBlock with advanced prompt engineering
    """

    def __init__(self, config: BaseFlowConfig):
        """Initialize flow with configuration.

        Args:
            config: Flow configuration with prompt engineering settings
        """
        self.config = config

        # Create LLMBlock directly from the embedded LLMConfig
        try:
            self.llm_block = create_llm_block(config.llm_config)
            logger.info(
                f"Initialized flow with model: {config.llm_config.model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM block: {e}")
            self.llm_block = None

    @property
    def client(self) -> Optional[Any]:
        """Get LLM client (for backward compatibility)."""
        return self.llm_block

    def build_prompt(self, knowledge_item: KnowledgeItem, **kwargs) -> str:
        """Build prompt for LLM processing using the enhanced template system.

        This method provides a flexible prompt building system:
        1. If custom_build_prompt function is provided, use it
        2. Otherwise, use template-based approach with variable substitution
        3. Support for both built-in and custom template variables

        Args:
            knowledge_item: KnowledgeItem object containing content
            **kwargs: Additional context or parameters

        Returns:
            Formatted prompt string ready for LLM processing
        """
        try:
            # Method 1: Use custom build function if provided
            if (
                hasattr(self.config, "custom_build_prompt")
                and self.config.custom_build_prompt
                and callable(self.config.custom_build_prompt)
            ):
                logger.debug("Using custom build_prompt function")
                return self.config.custom_build_prompt(knowledge_item, **kwargs)

            # Method 2: Use template-based approach
            template = (
                self.config.prompt_template
                or self.config.get_default_prompt_template()
            )

            # Extract template variables
            variables = self.config.extract_template_variables(
                knowledge_item, **kwargs
            )

            # Substitute variables in template
            prompt = self.config.substitute_template(template, variables)

            logger.debug(f"Built prompt using template (length: {len(prompt)})")
            return prompt

        except Exception as e:
            logger.error(f"Error building prompt: {e}")
            # Fallback to simple prompt
            return self._build_fallback_prompt(knowledge_item, **kwargs)

    def _build_fallback_prompt(
        self, knowledge_item: KnowledgeItem, **kwargs
    ) -> str:
        """Fallback prompt builder if template system fails.

        Args:
            knowledge_item: KnowledgeItem object
            **kwargs: Additional parameters

        Returns:
            Simple prompt string
        """
        system_prompt = self.config.get_system_prompt()
        content_summary = (
            f"Title: {knowledge_item.title}\n"
            f"Abstract: {knowledge_item.abstract or 'Not available'}\n"
            f"Content Type: {getattr(knowledge_item, 'content_type', 'generic')}"
        )

        return f"{system_prompt}\n\n{content_summary}\n\n{self.config.llm_config.custom_instructions or ''}"

    @abstractmethod
    def execute(self, knowledge_item: KnowledgeItem, **kwargs) -> Any:
        """Execute flow on knowledge item content.

        Args:
            knowledge_item: KnowledgeItem object to process
            **kwargs: Additional parameters

        Returns:
            Processed results (structure depends on flow)
        """
        pass

    def _call_llm(self, prompt: str, **kwargs) -> Optional[str]:
        """Call LLM with prompt using the LLMBlock.

        Args:
            prompt: Prompt to send to LLM
            **kwargs: Additional parameters to override config

        Returns:
            LLM response or None if failed
        """
        if not self.llm_block:
            logger.error("No LLM block available")
            return None

        try:
            response = self.llm_block.generate_text(prompt, **kwargs)
            return response
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None

    def test_connection(self) -> bool:
        """Test if the LLM connection is working.

        Returns:
            True if connection is working, False otherwise
        """
        if not self.llm_block:
            return False

        return self.llm_block.test_connection()

    def get_model_info(self) -> dict:
        """Get information about the current model.

        Returns:
            Model information dictionary
        """
        if not self.llm_block:
            return {"error": "No LLM block available"}

        return self.llm_block.get_info()

    def get_prompt_preview(
        self, knowledge_item: KnowledgeItem, **kwargs
    ) -> str:
        """Get a preview of the prompt that would be generated.

        Useful for debugging and prompt development.

        Args:
            knowledge_item: KnowledgeItem object
            **kwargs: Additional parameters

        Returns:
            Generated prompt string
        """
        return self.build_prompt(knowledge_item, **kwargs)

    def get_template_variables(
        self, knowledge_item: KnowledgeItem, **kwargs
    ) -> dict:
        """Get the variables that would be used for template substitution.

        Useful for debugging template issues.

        Args:
            knowledge_item: KnowledgeItem object
            **kwargs: Additional parameters

        Returns:
            Dictionary of template variables
        """
        return self.config.extract_template_variables(knowledge_item, **kwargs)

    def validate_template(self) -> tuple[bool, Optional[str]]:
        """Validate the current prompt template.

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not self.config.prompt_template:
                return True, None

            # Basic validation - check for balanced braces
            template = self.config.prompt_template
            if template.count("{{") != template.count("}}"):
                return False, "Unbalanced template braces"

            # Try to extract variable names
            import re

            variables = re.findall(r"\{\{([^}]+)\}\}", template)
            logger.debug(f"Template variables found: {variables}")

            return True, None
        except Exception as e:
            return False, str(e)
