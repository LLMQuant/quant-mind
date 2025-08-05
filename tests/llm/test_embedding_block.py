"""Tests for EmbeddingBlock."""

import unittest
from unittest import mock
from unittest.mock import Mock, patch

from quantmind.config import EmbeddingConfig
from quantmind.llm import EmbeddingBlock, create_embedding_block


class TestEmbeddingBlock(unittest.TestCase):
    """Test cases for EmbeddingBlock."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EmbeddingConfig(
            model="text-embedding-ada-002",
            api_key="test-key",
            timeout=30,
        )

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    def test_init_success(self, mock_litellm):
        """Test successful initialization."""
        block = EmbeddingBlock(self.config)

        self.assertEqual(block.config, self.config)
        mock_litellm.set_verbose = False
        self.assertEqual(mock_litellm.num_retries, 3)
        self.assertEqual(mock_litellm.request_timeout, 30)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", False)
    def test_init_litellm_unavailable(self):
        """Test initialization when LiteLLM is not available."""
        with self.assertRaises(ImportError) as context:
            EmbeddingBlock(self.config)

        self.assertIn(
            "litellm is required for EmbeddingBlock", str(context.exception)
        )

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("os.environ", {})
    def test_setup_litellm_openai(self, mock_litellm):
        """Test LiteLLM setup for OpenAI."""
        config = EmbeddingConfig(
            model="text-embedding-ada-002", api_key="test-key"
        )

        with patch("os.environ", {}) as mock_env:
            block = EmbeddingBlock(config)
            self.assertEqual(mock_env.get("OPENAI_API_KEY"), "test-key")

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("os.environ", {})
    def test_setup_litellm_azure(self, mock_litellm):
        """Test LiteLLM setup for Azure."""
        config = EmbeddingConfig(
            model="azure/text-embedding-ada-002", api_key="azure-key"
        )

        with patch("os.environ", {}) as mock_env:
            block = EmbeddingBlock(config)
            self.assertEqual(mock_env.get("AZURE_API_KEY"), "azure-key")

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_generate_embedding_success(self, mock_embedding, mock_litellm):
        """Test successful single embedding generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        result = block.generate_embedding("Test text")

        self.assertEqual(result, [0.1, 0.2, 0.3, 0.4, 0.5])
        mock_embedding.assert_called_once()

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_generate_embedding_failure(self, mock_embedding, mock_litellm):
        """Test embedding generation failure."""
        mock_embedding.side_effect = Exception("API Error")

        block = EmbeddingBlock(self.config)
        result = block.generate_embedding("Test text")

        self.assertIsNone(result)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_generate_embeddings_success(self, mock_embedding, mock_litellm):
        """Test successful multiple embedding generation."""
        # Mock response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6]),
        ]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        result = block.generate_embeddings(["Text 1", "Text 2"])

        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_embedding.assert_called_once()

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_generate_embeddings_failure(self, mock_embedding, mock_litellm):
        """Test multiple embedding generation failure."""
        mock_embedding.side_effect = Exception("API Error")

        block = EmbeddingBlock(self.config)
        result = block.generate_embeddings(["Text 1", "Text 2"])

        self.assertIsNone(result)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_call_with_retry_success(self, mock_embedding, mock_litellm):
        """Test successful call with retry."""
        mock_response = Mock()
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        result = block._call_with_retry(
            {"model": "text-embedding-ada-002", "input": "test"}
        )

        self.assertEqual(result, mock_response)
        mock_embedding.assert_called_once()

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    @patch("time.sleep")
    def test_call_with_retry_failure_then_success(
        self, mock_sleep, mock_embedding, mock_litellm
    ):
        """Test retry logic with failure then success."""
        mock_response = Mock()
        mock_embedding.side_effect = [
            Exception("First failure"),
            mock_response,
        ]

        block = EmbeddingBlock(self.config)
        result = block._call_with_retry(
            {"model": "text-embedding-ada-002", "input": "test"}
        )

        self.assertEqual(result, mock_response)
        self.assertEqual(mock_embedding.call_count, 2)
        mock_sleep.assert_called_once_with(1.0)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    @patch("time.sleep")
    def test_call_with_retry_all_failures(
        self, mock_sleep, mock_embedding, mock_litellm
    ):
        """Test retry logic with all failures."""
        mock_embedding.side_effect = Exception("Always fails")

        block = EmbeddingBlock(self.config)
        result = block._call_with_retry(
            {"model": "text-embedding-ada-002", "input": "test"}
        )

        self.assertIsNone(result)
        self.assertEqual(mock_embedding.call_count, 4)  # 1 initial + 3 retries
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_test_connection_success(self, mock_embedding, mock_litellm):
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        result = block.test_connection()

        self.assertTrue(result)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_test_connection_failure(self, mock_embedding, mock_litellm):
        """Test connection test failure."""
        mock_embedding.side_effect = Exception("Connection failed")

        block = EmbeddingBlock(self.config)
        result = block.test_connection()

        self.assertFalse(result)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_get_embedding_dimension_from_config(
        self, mock_embedding, mock_litellm
    ):
        """Test getting embedding dimension from config."""
        config = EmbeddingConfig(
            model="text-embedding-3-small",
            dimensions=512,
        )
        block = EmbeddingBlock(config)
        dimension = block.get_embedding_dimension()

        self.assertEqual(dimension, 512)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_get_embedding_dimension_from_test_embedding(
        self, mock_embedding, mock_litellm
    ):
        """Test getting embedding dimension by generating test embedding."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        dimension = block.get_embedding_dimension()

        self.assertEqual(dimension, 5)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_get_embedding_dimension_failure(
        self, mock_embedding, mock_litellm
    ):
        """Test getting embedding dimension when test embedding fails."""
        mock_embedding.side_effect = Exception("API Error")

        block = EmbeddingBlock(self.config)
        dimension = block.get_embedding_dimension()

        self.assertIsNone(dimension)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    def test_get_info(self, mock_litellm):
        """Test getting block info."""
        block = EmbeddingBlock(self.config)
        info = block.get_info()

        expected_keys = ["model", "provider", "timeout", "retry_attempts"]
        for key in expected_keys:
            self.assertIn(key, info)

        self.assertEqual(info["model"], "text-embedding-ada-002")
        self.assertEqual(info["provider"], "openai")
        self.assertEqual(info["timeout"], 30)
        self.assertEqual(info["retry_attempts"], 3)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    def test_update_config(self, mock_litellm):
        """Test configuration update."""
        block = EmbeddingBlock(self.config)

        # Check initial config
        self.assertEqual(block.config.timeout, 30)
        self.assertEqual(block.config.api_key, "test-key")

        # Update config
        block.update_config(timeout=60, api_key="new-key")

        # Check updated config
        self.assertEqual(block.config.timeout, 60)
        self.assertEqual(block.config.api_key, "new-key")
        # Other values should remain unchanged
        self.assertEqual(block.config.model, "text-embedding-ada-002")

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    def test_temporary_config(self, mock_litellm):
        """Test temporary configuration context manager."""
        block = EmbeddingBlock(self.config)

        # Check initial config
        self.assertEqual(block.config.timeout, 30)

        # Use temporary config
        with block.temporary_config(timeout=60):
            self.assertEqual(block.config.timeout, 60)

        # Check config is restored
        self.assertEqual(block.config.timeout, 30)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_batch_embed_success(self, mock_embedding, mock_litellm):
        """Test successful batch embedding."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
        ]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        result = block.batch_embed(texts, batch_size=2)

        expected = [[0.1, 0.2], [0.3, 0.4], [0.1, 0.2], [0.3, 0.4]]
        self.assertEqual(result, expected)
        self.assertEqual(mock_embedding.call_count, 2)  # 2 batches

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_batch_embed_failure(self, mock_embedding, mock_litellm):
        """Test batch embedding failure."""
        mock_embedding.side_effect = Exception("API Error")

        block = EmbeddingBlock(self.config)
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        result = block.batch_embed(texts, batch_size=2)

        self.assertIsNone(result)

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    @patch("time.sleep")
    def test_batch_embed_with_delay(
        self, mock_sleep, mock_embedding, mock_litellm
    ):
        """Test batch embedding with delay between batches."""
        config = EmbeddingConfig(
            model="text-embedding-ada-002",
            api_key="test-key",
            timeout=30,
            retry_delay=0.1,
        )

        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
        ]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(config)
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        result = block.batch_embed(texts, batch_size=2)

        # Should have delay between batches
        self.assertEqual(mock_sleep.call_count, 1)  # Delay between 2 batches

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_generate_embedding_with_kwargs(self, mock_embedding, mock_litellm):
        """Test embedding generation with additional kwargs."""
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = [0.1, 0.2, 0.3]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        result = block.generate_embedding(
            "Test text", dimensions=512, user="test_user"
        )

        # Check that kwargs were passed to the embedding call
        call_args = mock_embedding.call_args
        self.assertIn("dimensions", call_args[1])
        self.assertIn("user", call_args[1])
        self.assertEqual(call_args[1]["dimensions"], 512)
        self.assertEqual(call_args[1]["user"], "test_user")

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    @patch("quantmind.llm.embedding.embedding")
    def test_generate_embeddings_with_kwargs(
        self, mock_embedding, mock_litellm
    ):
        """Test multiple embedding generation with additional kwargs."""
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
        ]
        mock_embedding.return_value = mock_response

        block = EmbeddingBlock(self.config)
        result = block.generate_embeddings(
            ["Text 1", "Text 2"], dimensions=512, user="test_user"
        )

        # Check that kwargs were passed to the embedding call
        call_args = mock_embedding.call_args
        self.assertIn("dimensions", call_args[1])
        self.assertIn("user", call_args[1])
        self.assertEqual(call_args[1]["dimensions"], 512)
        self.assertEqual(call_args[1]["user"], "test_user")


class TestCreateEmbeddingBlock(unittest.TestCase):
    """Test cases for create_embedding_block function."""

    @patch("quantmind.llm.embedding.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.embedding.litellm")
    def test_create_embedding_block(self, mock_litellm):
        """Test EmbeddingBlock creation."""
        config = EmbeddingConfig(model="text-embedding-ada-002")
        block = create_embedding_block(config)

        self.assertIsInstance(block, EmbeddingBlock)
        self.assertEqual(block.config, config)


if __name__ == "__main__":
    unittest.main()
