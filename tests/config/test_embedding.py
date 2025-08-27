"""Tests for Embedding configuration."""

import unittest
from unittest.mock import patch

from quantmind.config.embedding import EmbeddingConfig


class TestEmbeddingConfig(unittest.TestCase):
    """Test cases for EmbeddingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        # Test default values
        self.assertEqual(config.model, "text-embedding-ada-002")
        self.assertIsNone(config.user)
        self.assertIsNone(config.dimensions)
        self.assertEqual(config.encoding_format, "float")
        self.assertEqual(config.timeout, 600)
        self.assertIsNone(config.api_base)
        self.assertIsNone(config.api_version)
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.api_type)

    def test_custom_config(self):
        """Test custom configuration values."""
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            user="test_user_123",
            dimensions=512,
            encoding_format="base64",
            timeout=1,
            api_key="test-key",
            api_base="https://api.example.com",
            api_version="2023-05-15",
            api_type="azure",
        )

        self.assertEqual(config.model, "text-embedding-3-small")
        self.assertEqual(config.user, "test_user_123")
        self.assertEqual(config.dimensions, 512)
        self.assertEqual(config.encoding_format, "base64")
        self.assertEqual(config.timeout, 1)
        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.api_base, "https://api.example.com")
        self.assertEqual(config.api_version, "2023-05-15")
        self.assertEqual(config.api_type, "azure")

    def test_validation_model(self):
        """Test model validation."""
        # Valid model
        config = EmbeddingConfig(model="text-embedding-ada-002")
        self.assertEqual(config.model, "text-embedding-ada-002")

        # Empty model should raise error
        with self.assertRaises(ValueError):
            EmbeddingConfig(model="")

        # None model should raise error
        with self.assertRaises(ValueError):
            EmbeddingConfig(model=None)

        # Whitespace should be stripped
        config = EmbeddingConfig(model="  text-embedding-ada-002  ")
        self.assertEqual(config.model, "text-embedding-ada-002")

    def test_validation_api_key(self):
        """Test API key validation."""
        # Valid API key
        config = EmbeddingConfig(api_key="test-key")
        self.assertEqual(config.api_key, "test-key")

        # None API key is valid
        config = EmbeddingConfig(api_key=None)
        self.assertIsNone(config.api_key)

        # Invalid API key type should raise error
        with self.assertRaises(ValueError):
            EmbeddingConfig(api_key=123)

        with self.assertRaises(ValueError):
            EmbeddingConfig(api_key=[])

    def test_get_provider_type(self):
        """Test provider type detection."""
        # OpenAI models
        config = EmbeddingConfig(model="text-embedding-ada-002")
        self.assertEqual(config.get_provider_type(), "openai")

        config = EmbeddingConfig(model="text-embedding-3-small")
        self.assertEqual(config.get_provider_type(), "openai")

        config = EmbeddingConfig(model="text-embedding-3-large")
        self.assertEqual(config.get_provider_type(), "openai")

        # Azure models
        config = EmbeddingConfig(model="azure/text-embedding-ada-002")
        self.assertEqual(config.get_provider_type(), "azure")

        config = EmbeddingConfig(model="text-embedding-ada-002-azure")
        self.assertEqual(config.get_provider_type(), "azure")

        # Gemini models
        config = EmbeddingConfig(model="gemini/embed-multilingual-v3.0")
        self.assertEqual(config.get_provider_type(), "gemini")

        # Unknown models
        config = EmbeddingConfig(model="unknown-model")
        self.assertEqual(config.get_provider_type(), "unknown")

    def test_get_litellm_params_minimal(self):
        """Test get_litellm_params with minimal configuration."""
        config = EmbeddingConfig(model="text-embedding-ada-002")
        params = config.get_litellm_params()

        self.assertEqual(params["model"], "text-embedding-ada-002")
        self.assertIn("encoding_format", params)
        self.assertEqual(len(params), 2)  # Only model and encoding_format

    def test_get_litellm_params_full(self):
        """Test get_litellm_params with full configuration."""
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            user="test_user",
            dimensions=512,
            encoding_format="base64",
            timeout=1,
            api_key="test-key",
            api_base="https://api.example.com",
            api_version="2023-05-15",
            api_type="azure",
        )
        params = config.get_litellm_params()

        expected_params = {
            "model": "text-embedding-3-small",
            "user": "test_user",
            "dimensions": 512,
            "encoding_format": "base64",
            "api_base": "https://api.example.com",
            "api_version": "2023-05-15",
            "api_key": "test-key",
            "api_type": "azure",
        }

        self.assertEqual(params, expected_params)

    def test_get_litellm_params_partial(self):
        """Test get_litellm_params with partial configuration."""
        config = EmbeddingConfig(
            model="text-embedding-ada-002",
            user="test_user",
            dimensions=1536,
            api_key="test-key",
        )
        params = config.get_litellm_params()

        expected_params = {
            "model": "text-embedding-ada-002",
            "user": "test_user",
            "dimensions": 1536,
            "encoding_format": "float",
            "api_key": "test-key",
        }

        self.assertEqual(params, expected_params)

    def test_create_variant(self):
        """Test creating configuration variants."""
        base_config = EmbeddingConfig(
            model="text-embedding-ada-002",
            timeout=1,
            api_key="base-key",
        )

        # Create variant with overrides
        variant = base_config.create_variant(
            timeout=1,
            api_key="variant-key",
            user="test_user",
        )

        # Original config should be unchanged
        self.assertEqual(base_config.timeout, 1)
        self.assertEqual(base_config.api_key, "base-key")
        self.assertIsNone(base_config.user)

        # Variant should have new values
        self.assertEqual(variant.timeout, 1)
        self.assertEqual(variant.api_key, "variant-key")
        self.assertEqual(variant.user, "test_user")
        self.assertEqual(variant.model, "text-embedding-ada-002")  # Unchanged

    def test_create_variant_empty(self):
        """Test creating variant with no overrides."""
        base_config = EmbeddingConfig(
            model="text-embedding-ada-002",
            timeout=1,
        )

        variant = base_config.create_variant()

        # Should be identical to base config
        self.assertEqual(variant.model, base_config.model)
        self.assertEqual(variant.timeout, base_config.timeout)
        self.assertEqual(variant.encoding_format, base_config.encoding_format)

    def test_encoding_format_validation(self):
        """Test encoding format validation."""
        # Valid encoding formats
        config = EmbeddingConfig(encoding_format="float")
        self.assertEqual(config.encoding_format, "float")

        config = EmbeddingConfig(encoding_format="base64")
        self.assertEqual(config.encoding_format, "base64")

    def test_dimensions_validation(self):
        """Test dimensions validation."""
        # Valid dimensions
        config = EmbeddingConfig(dimensions=512)
        self.assertEqual(config.dimensions, 512)

        config = EmbeddingConfig(dimensions=1536)
        self.assertEqual(config.dimensions, 1536)

        config = EmbeddingConfig(dimensions=3072)
        self.assertEqual(config.dimensions, 3072)

        # None is valid
        config = EmbeddingConfig(dimensions=None)
        self.assertIsNone(config.dimensions)

        # Zero and negative dimensions should be allowed (validation handled by API)
        config = EmbeddingConfig(dimensions=0)
        self.assertEqual(config.dimensions, 0)

        config = EmbeddingConfig(dimensions=-1)
        self.assertEqual(config.dimensions, -1)

    def test_timeout_validation(self):
        """Test timeout validation."""
        # Valid timeouts
        config = EmbeddingConfig(timeout=1)
        self.assertEqual(config.timeout, 1)

        config = EmbeddingConfig(timeout=1)
        self.assertEqual(config.timeout, 1)

        config = EmbeddingConfig(timeout=1)
        self.assertEqual(config.timeout, 1)

        # Zero and negative timeouts should be allowed (validation handled by API)
        config = EmbeddingConfig(timeout=0)
        self.assertEqual(config.timeout, 0)

        config = EmbeddingConfig(timeout=-1)
        self.assertEqual(config.timeout, -1)

    def test_equality(self):
        """Test config equality."""
        config1 = EmbeddingConfig(
            model="text-embedding-ada-002",
            user="test_user",
            dimensions=512,
        )

        config2 = EmbeddingConfig(
            model="text-embedding-ada-002",
            user="test_user",
            dimensions=512,
        )

        config3 = EmbeddingConfig(
            model="text-embedding-3-small",
            user="test_user",
            dimensions=512,
        )

        self.assertEqual(config1, config2)
        self.assertNotEqual(config1, config3)

    def test_repr(self):
        """Test config string representation."""
        config = EmbeddingConfig(
            model="text-embedding-ada-002",
            user="test_user",
            dimensions=512,
        )

        repr_str = repr(config)
        self.assertIn("text-embedding-ada-002", repr_str)
        self.assertIn("test_user", repr_str)
        self.assertIn("512", repr_str)

    def test_str(self):
        """Test config string representation."""
        config = EmbeddingConfig(
            model="text-embedding-ada-002",
            user="test_user",
            dimensions=512,
        )

        str_repr = str(config)
        self.assertIn("text-embedding-ada-002", str_repr)
        self.assertIn("test_user", str_repr)
        self.assertIn("512", str_repr)


if __name__ == "__main__":
    unittest.main()
