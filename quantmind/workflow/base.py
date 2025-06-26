"""Base workflow abstract class for QuantMind framework."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from quantmind.models.paper import Paper
from quantmind.utils.logger import get_logger
from quantmind.config.workflows import BaseWorkflowConfig

logger = get_logger(__name__)


class BaseWorkflow(ABC):
    """Abstract base class for all LLM-based workflows.

    All workflows follow the pattern:
    1. build_prompt() - Create prompt from content and configuration
    2. execute() - Process content using LLM and return structured results
    """

    def __init__(self, config: BaseWorkflowConfig):
        """Initialize workflow with configuration.

        Args:
            config: Workflow configuration
        """
        self.config = config
        self.client = None
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the LLM client based on configuration."""
        try:
            if self.config.llm_type.lower() == "openai":
                self._init_openai_client()
            elif self.config.llm_type.lower() == "camel":
                self._init_camel_client()
            else:
                logger.warning(f"Unsupported LLM type: {self.config.llm_type}")
                self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.client = None

    def _init_openai_client(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai

            client_kwargs = {
                "api_key": self.config.api_key,
                "timeout": self.config.timeout,
            }

            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url

            self.client = openai.OpenAI(**client_kwargs)
            logger.info("Initialized OpenAI client")

        except ImportError:
            logger.error("OpenAI library not available")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def _init_camel_client(self) -> None:
        """Initialize CAMEL client."""
        try:
            from camel.agents import ChatAgent
            from camel.models import ModelFactory
            from camel.types import ModelPlatformType, ModelType

            model_instance = ModelFactory.create(
                model_platform=ModelPlatformType.OPENAI,
                model_type=ModelType.GPT_4O,
                model_config_dict={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
            self.client = ChatAgent(model=model_instance)
            logger.info("Initialized CAMEL client")

        except ImportError:
            logger.error("CAMEL library not available")
            self.client = None
        except Exception as e:
            logger.error(f"Failed to initialize CAMEL client: {e}")
            self.client = None

    @abstractmethod
    def build_prompt(self, paper: Paper, **kwargs) -> str:
        """Build prompt for LLM processing.

        Args:
            paper: Paper object containing content
            **kwargs: Additional context or parameters

        Returns:
            Formatted prompt string
        """
        pass

    @abstractmethod
    def execute(self, paper: Paper, **kwargs) -> Any:
        """Execute workflow on paper content.

        Args:
            paper: Paper object to process
            **kwargs: Additional parameters

        Returns:
            Processed results (structure depends on workflow)
        """
        pass

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call LLM with prompt and handle retries.

        Args:
            prompt: Prompt to send to LLM

        Returns:
            LLM response or None if failed
        """
        if not self.client:
            logger.error("No LLM client available")
            return None

        for attempt in range(self.config.retry_attempts + 1):
            try:
                if self.config.llm_type.lower() == "openai":
                    return self._call_openai(prompt)
                elif self.config.llm_type.lower() == "camel":
                    return self._call_camel(prompt)
                else:
                    logger.error(
                        f"Unsupported LLM type: {self.config.llm_type}"
                    )
                    return None

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.config.retry_attempts:
                    logger.error(f"All LLM call attempts failed")
                    return None

        return None

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API.

        Args:
            prompt: Prompt to send

        Returns:
            Response content
        """
        response = self.client.chat.completions.create(
            model=self.config.llm_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def _call_camel(self, prompt: str) -> str:
        """Call CAMEL API.

        Args:
            prompt: Prompt to send

        Returns:
            Response content
        """
        response = self.client.step(prompt)
        return response.msgs[0].content.strip()

    def _append_custom_instructions(self, prompt: str) -> str:
        """Append custom instructions to prompt if configured.

        Args:
            prompt: Base prompt

        Returns:
            Prompt with custom instructions appended
        """
        if self.config.custom_instructions:
            return f"{prompt}\n\nAdditional Instructions:\n{self.config.custom_instructions}"
        return prompt
