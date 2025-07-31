"""Tests for EmbeddingBlock."""

import unittest
from unittest.mock import Mock, patch

from quantmind.config import EmbeddingConfig
from quantmind.llm import EmbeddingBlock, create_embedding_block


class TestEmbeddingConfig(unittest.TestCase):
    """Test EmbeddingConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = EmbeddingConfig()

        self.assertEqual(config.model, "text-embedding-ada-002")
        self.assertIsNone(config.api_key)
        self.assertEqual(config.timeout, 600)


if __name__ == "__main__":
    unittest.main()
