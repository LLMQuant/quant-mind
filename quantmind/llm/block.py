"""LLMBlock - A reusable LLM function block using LiteLLM."""

import os
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from quantmind.utils.logger import get_logger

from ..config import LLMConfig

logger = get_logger(__name__)

try:
    import litellm
    from litellm import completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


class LLMBlock:
    """A reusable LLM function block using LiteLLM.

    LLMBlock provides a consistent interface for LLM operations across
    different providers (OpenAI, Anthropic, Google, Azure, etc.).

    Unlike workflows, LLMBlock focuses on providing basic LLM capabilities
    without business logic.
    """

    def __init__(self, config: LLMConfig):
        """Initialize LLMBlock with configuration.

        Args:
            config: LLM configuration

        Raises:
            ImportError: If LiteLLM is not available
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not available. Please install it with: pip install litellm"
            )

        self.config = config
        self._setup_litellm()

        logger.info(f"Initialized LLMBlock with model: {config.model}")

    def _setup_litellm(self):
        """Setup LiteLLM configuration."""
        # Set global LiteLLM settings
        litellm.set_verbose = False  # Disable verbose logging by default

        # Configure retries
        litellm.num_retries = self.config.retry_attempts
        litellm.request_timeout = self.config.timeout

        # Set API key as environment variable if provided
        if self.config.api_key:
            provider_type = self.config.get_provider_type()
            if provider_type == "openai":
                os.environ["OPENAI_API_KEY"] = self.config.api_key
            elif provider_type == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = self.config.api_key
            elif provider_type == "google":
                os.environ["GOOGLE_API_KEY"] = self.config.api_key
            elif provider_type == "deepseek":
                os.environ["DEEPSEEK_API_KEY"] = self.config.api_key

        logger.debug(
            f"Configured LiteLLM for provider: {self.config.get_provider_type()}"
        )

    def generate_text(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> Optional[str]:
        """Generate text using the configured LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (overrides config)
            **kwargs: Additional parameters to override config

        Returns:
            Generated text or None if failed
        """
        try:
            # Build messages
            messages = self._build_messages(prompt, system_prompt)

            # Get LiteLLM parameters
            params = self.config.get_litellm_params()
            params.update(kwargs)  # Allow runtime overrides

            # Add messages to parameters
            params["messages"] = messages

            # Call LiteLLM with retry logic
            response = self._call_with_retry(params)

            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    return content.strip()

            logger.warning("No content received from LLM")
            return None

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return None

    def generate_structured_output(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Generate structured output (JSON) using the configured LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: JSON schema for structured output
            **kwargs: Additional parameters

        Returns:
            Parsed JSON response or None if failed
        """
        try:
            # Build messages
            messages = self._build_messages(prompt, system_prompt)

            # Get LiteLLM parameters
            params = self.config.get_litellm_params()
            params.update(kwargs)
            params["messages"] = messages

            # Add response format if provided
            # TODO: Refactor the response_format to be more generic
            if response_format:
                provider_type = self.config.get_provider_type()
                if provider_type == "openai":
                    params["response_format"] = response_format
                elif (
                    provider_type == "google"
                    and "response_schema" in response_format
                ):
                    # Gemini specific format
                    params["response_format"] = response_format

            # Call LiteLLM
            response = self._call_with_retry(params)

            if response and response.choices:
                content = response.choices[0].message.content
                if content:
                    # Try to parse JSON
                    import json

                    try:
                        return json.loads(content.strip())
                    except json.JSONDecodeError:
                        # Fallback: try to extract JSON from text
                        return self._extract_json_from_text(content)

            return None

        except Exception as e:
            logger.error(f"Error generating structured output: {e}")
            return None

    def _build_messages(
        self, prompt: str, system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Build messages array for LLM call.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Messages array
        """
        messages = []

        # Add system prompt
        final_system_prompt = system_prompt or self.config.system_prompt
        if final_system_prompt:
            messages.append({"role": "system", "content": final_system_prompt})

        # Add user prompt with custom instructions
        final_prompt = prompt
        if self.config.custom_instructions:
            final_prompt = f"{prompt}\n\nAdditional Instructions:\n{self.config.custom_instructions}"

        messages.append({"role": "user", "content": final_prompt})

        return messages

    def _call_with_retry(self, params: Dict[str, Any]) -> Optional[Any]:
        """Call LiteLLM with retry logic.

        Args:
            params: LiteLLM parameters

        Returns:
            LiteLLM response or None
        """
        last_exception = None

        for attempt in range(self.config.retry_attempts + 1):
            try:
                logger.debug(
                    f"LLM call attempt {attempt + 1}/{self.config.retry_attempts + 1}"
                )

                response = completion(**params)

                # Log usage if available
                if hasattr(response, "usage") and response.usage:
                    logger.debug(f"Token usage: {response.usage}")

                return response

            except Exception as e:
                last_exception = e
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")

                if attempt < self.config.retry_attempts:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(
                        f"All {self.config.retry_attempts + 1} attempts failed"
                    )

        # Log final error
        if last_exception:
            logger.error(f"Final error: {last_exception}")

        return None

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text response as fallback.

        Args:
            text: Response text

        Returns:
            Parsed JSON or None
        """
        import json
        import re

        # Try to find JSON objects in the text
        json_patterns = [
            r"\{[^{}]*\}",  # Simple JSON object
            r"\{.*?\}",  # JSON object with nested content
            r"\[.*?\]",  # JSON array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        logger.warning("Could not extract JSON from text response")
        return None

    def test_connection(self) -> bool:
        """Test if the LLM connection is working.

        Returns:
            True if connection is working, False otherwise
        """
        try:
            response = self.generate_text(
                "Hello, this is a test. Please respond with 'OK'."
            )
            return response is not None and len(response) > 0
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get information about the current LLMBlock.

        Returns:
            LLMBlock information dictionary
        """
        return {
            "model": self.config.model,
            "provider": self.config.get_provider_type(),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "retry_attempts": self.config.retry_attempts,
        }

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        # Create new config with overrides
        self.config = self.config.create_variant(**kwargs)

        # Re-setup LiteLLM
        self._setup_litellm()

        logger.info(f"Updated LLMBlock configuration")

    @contextmanager
    def temporary_config(self, **kwargs):
        """Context manager for temporary configuration changes.

        Args:
            **kwargs: Temporary configuration overrides
        """
        original_config = self.config.model_copy()

        try:
            self.update_config(**kwargs)
            yield self
        finally:
            self.config = original_config
            self._setup_litellm()


def create_llm_block(config: LLMConfig) -> LLMBlock:
    """Create a new LLMBlock instance.

    Args:
        config: LLM configuration

    Returns:
        New LLMBlock instance
    """
    return LLMBlock(config)
