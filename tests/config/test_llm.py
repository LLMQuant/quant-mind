"""Tests for LLM configuration."""

import unittest
from unittest.mock import patch

from quantmind.config.llm import LLMConfig


class TestLLMConfig(unittest.TestCase):
    """Test cases for LLMConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()

        # Test default values
        self.assertEqual(config.model, "gpt-4o")
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.max_tokens, 4000)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.retry_attempts, 3)
        self.assertEqual(config.retry_delay, 1.0)
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.base_url)
        self.assertIsNone(config.system_prompt)
        self.assertEqual(config.extra_params, {})

    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMConfig(
            model="claude-3-5-sonnet-20241022",
            temperature=0.7,
            max_tokens=2000,
            api_key="test-key",
            base_url="https://api.example.com",
            system_prompt="You are a helpful assistant.",
            extra_params={"frequency_penalty": 0.1},
        )

        self.assertEqual(config.model, "claude-3-5-sonnet-20241022")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 2000)
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://api.example.com")
        self.assertEqual(config.system_prompt, "You are a helpful assistant.")
        self.assertEqual(config.extra_params, {"frequency_penalty": 0.1})

    def test_validation_model(self):
        """Test model validation."""
        # Valid model
        config = LLMConfig(model="gpt-4o")
        self.assertEqual(config.model, "gpt-4o")

        # Empty model should raise error
        with self.assertRaises(ValueError):
            LLMConfig(model="")

        # Whitespace should be stripped
        config = LLMConfig(model="  gpt-4o  ")
        self.assertEqual(config.model, "gpt-4o")

    def test_validation_temperature(self):
        """Test temperature validation."""
        # Valid temperatures
        LLMConfig(temperature=0.0)
        LLMConfig(temperature=1.0)
        LLMConfig(temperature=2.0)

        # Invalid temperatures
        with self.assertRaises(ValueError):
            LLMConfig(temperature=-0.1)

        with self.assertRaises(ValueError):
            LLMConfig(temperature=2.1)

    def test_validation_max_tokens(self):
        """Test max_tokens validation."""
        # Valid max_tokens
        LLMConfig(max_tokens=1)
        LLMConfig(max_tokens=4000)

        # Invalid max_tokens
        with self.assertRaises(ValueError):
            LLMConfig(max_tokens=0)

        with self.assertRaises(ValueError):
            LLMConfig(max_tokens=-1)

    def test_get_provider_type(self):
        """Test provider type detection."""
        # OpenAI
        config = LLMConfig(model="gpt-4o")
        self.assertEqual(config.get_provider_type(), "openai")

        config = LLMConfig(model="openai/gpt-4o")
        self.assertEqual(config.get_provider_type(), "openai")

        # Anthropic
        config = LLMConfig(model="claude-3-5-sonnet-20241022")
        self.assertEqual(config.get_provider_type(), "anthropic")

        config = LLMConfig(model="anthropic/claude-3-5-sonnet")
        self.assertEqual(config.get_provider_type(), "anthropic")

        # Google
        config = LLMConfig(model="gemini-pro")
        self.assertEqual(config.get_provider_type(), "google")

        config = LLMConfig(model="google/gemini-pro")
        self.assertEqual(config.get_provider_type(), "google")

        # Azure
        config = LLMConfig(model="azure/gpt-4o")
        self.assertEqual(config.get_provider_type(), "azure")

        # Ollama
        config = LLMConfig(model="ollama/llama2")
        self.assertEqual(config.get_provider_type(), "ollama")

        # Unknown
        config = LLMConfig(model="unknown-model")
        self.assertEqual(config.get_provider_type(), "unknown")

    def test_get_litellm_params(self):
        """Test LiteLLM parameters generation."""
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=2000,
            api_key="test-key",
            base_url="https://api.example.com",
            extra_params={"frequency_penalty": 0.1},
        )

        params = config.get_litellm_params()

        expected_params = {
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0,
            "timeout": 60,
            "api_key": "test-key",
            "base_url": "https://api.example.com",
            "frequency_penalty": 0.1,
        }

        self.assertEqual(params, expected_params)

    def test_get_litellm_params_minimal(self):
        """Test LiteLLM parameters with minimal config."""
        config = LLMConfig()
        params = config.get_litellm_params()

        # Since we will automatically resolve the API key, we should remove it from the parameters.
        if "api_key" in params:
            params.pop("api_key")

        expected_params = {
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 4000,
            "top_p": 1.0,
            "timeout": 60,
        }

        self.assertEqual(params, expected_params)

    def test_create_variant(self):
        """Test creating configuration variants."""
        base_config = LLMConfig(
            model="gpt-4o", temperature=0.0, api_key="base-key"
        )

        # Create variant with overrides
        variant = base_config.create_variant(
            temperature=0.7, max_tokens=2000, api_key="variant-key"
        )

        # Check variant has overrides
        self.assertEqual(variant.temperature, 0.7)
        self.assertEqual(variant.max_tokens, 2000)
        self.assertEqual(variant.api_key, "variant-key")

        # Check variant keeps non-overridden values
        self.assertEqual(variant.model, "gpt-4o")

        # Check original config is unchanged
        self.assertEqual(base_config.temperature, 0.0)
        self.assertEqual(base_config.max_tokens, 4000)
        self.assertEqual(base_config.api_key, "base-key")

    def test_api_key_validation(self):
        """Test API key validation."""
        # Valid API key
        config = LLMConfig(api_key="test-key")
        self.assertEqual(config.api_key, "test-key")

        # None API key is valid
        config = LLMConfig(api_key=None)
        self.assertIsNone(config.api_key)

        # Non-string API key should raise error
        with self.assertRaises(ValueError):
            LLMConfig(api_key=123)


if __name__ == "__main__":
    unittest.main()
