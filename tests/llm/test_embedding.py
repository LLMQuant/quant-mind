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
        self.assertEqual(config.timeout, 60)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.normalize)

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            api_key="test_key",
            timeout=30,
            batch_size=16,
            normalize=False,
            device="cuda"
        )
        
        self.assertEqual(config.model, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.timeout, 30)
        self.assertEqual(config.batch_size, 16)
        self.assertFalse(config.normalize)
        self.assertEqual(config.device, "cuda")

    def test_provider_detection(self):
        """Test provider type detection."""
        # OpenAI models
        config = EmbeddingConfig(model="text-embedding-ada-002")
        self.assertEqual(config.get_provider_type(), "openai")
        
        config = EmbeddingConfig(model="text-embedding-3-small")
        self.assertEqual(config.get_provider_type(), "openai")
        
        # SentenceTransformers models
        config = EmbeddingConfig(model="sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(config.get_provider_type(), "sentence_transformers")
        
        config = EmbeddingConfig(model="all-MiniLM-L6-v2")
        self.assertEqual(config.get_provider_type(), "sentence_transformers")
        
        # Cohere models
        config = EmbeddingConfig(model="embed-english-v3.0")
        self.assertEqual(config.get_provider_type(), "cohere")
        
        config = EmbeddingConfig(model="cohere/embed-multilingual-v3.0")
        self.assertEqual(config.get_provider_type(), "cohere")

    def test_create_variant(self):
        """Test creating configuration variants."""
        base_config = EmbeddingConfig(
            model="text-embedding-ada-002",
            timeout=60,
            batch_size=32
        )
        
        variant = base_config.create_variant(
            timeout=30,
            batch_size=16
        )
        
        self.assertEqual(variant.model, "text-embedding-ada-002")
        self.assertEqual(variant.timeout, 30)
        self.assertEqual(variant.batch_size, 16)

    def test_validation(self):
        """Test configuration validation."""
        # Test invalid model
        with self.assertRaises(ValueError):
            EmbeddingConfig(model="")
        
        with self.assertRaises(ValueError):
            EmbeddingConfig(model=None)
        
        # Test invalid API key
        with self.assertRaises(ValueError):
            EmbeddingConfig(api_key=123)


class TestEmbeddingBlock(unittest.TestCase):
    """Test EmbeddingBlock class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2",
            normalize=True
        )

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_initialization(self, mock_sentence_transformer):
        """Test EmbeddingBlock initialization."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        
        self.assertEqual(embedding_block.config, self.config)
        mock_sentence_transformer.assert_called_once_with(self.config.model)

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', False)
    def test_initialization_missing_dependency(self):
        """Test initialization with missing dependency."""
        with self.assertRaises(ImportError):
            EmbeddingBlock(self.config)

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_generate_embedding(self, mock_sentence_transformer):
        """Test single embedding generation."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        embedding = embedding_block.generate_embedding("test text")
        
        self.assertEqual(embedding, [0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.encode.assert_called_once_with("test text")

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_generate_embeddings(self, mock_sentence_transformer):
        """Test multiple embedding generation."""
        mock_model = Mock()
        mock_model.encode.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        embeddings = embedding_block.generate_embeddings(["text1", "text2"])
        
        self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.assert_called_once_with(["text1", "text2"])

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_get_embedding_dimension(self, mock_sentence_transformer):
        """Test getting embedding dimension."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        dimension = embedding_block.get_embedding_dimension()
        
        self.assertEqual(dimension, 5)

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_test_connection(self, mock_sentence_transformer):
        """Test connection testing."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        result = embedding_block.test_connection()
        
        self.assertTrue(result)

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_get_info(self, mock_sentence_transformer):
        """Test getting embedding block information."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        info = embedding_block.get_info()
        
        self.assertEqual(info["model"], self.config.model)
        self.assertEqual(info["provider"], "sentence_transformers")
        self.assertEqual(info["dimension"], 3)
        self.assertIn("config", info)

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_update_config(self, mock_sentence_transformer):
        """Test configuration updates."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        
        # Update timeout
        embedding_block.update_config(timeout=30)
        self.assertEqual(embedding_block.config.timeout, 30)
        
        # Update model (should reinitialize)
        embedding_block.update_config(model="sentence-transformers/all-mpnet-base-v2")
        self.assertEqual(embedding_block.config.model, "sentence-transformers/all-mpnet-base-v2")

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_temporary_config(self, mock_sentence_transformer):
        """Test temporary configuration context manager."""
        mock_model = Mock()
        mock_model.encode.return_value = [0.1, 0.2, 0.3]
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        original_timeout = embedding_block.config.timeout
        
        with embedding_block.temporary_config(timeout=10):
            self.assertEqual(embedding_block.config.timeout, 10)
        
        # Should be restored
        self.assertEqual(embedding_block.config.timeout, original_timeout)

    @patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('quantmind.llm.embedding.SentenceTransformer')
    def test_batch_embed(self, mock_sentence_transformer):
        """Test batch embedding processing."""
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_sentence_transformer.return_value = mock_model
        
        embedding_block = EmbeddingBlock(self.config)
        texts = ["text1", "text2", "text3", "text4"]
        
        embeddings = embedding_block.batch_embed(texts, batch_size=2)
        
        self.assertEqual(len(embeddings), 4)
        # Should have been called twice (2 batches of 2)
        self.assertEqual(mock_model.encode.call_count, 2)


class TestCreateEmbeddingBlock(unittest.TestCase):
    """Test create_embedding_block function."""

    def test_create_embedding_block(self):
        """Test creating embedding block."""
        config = EmbeddingConfig(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        with patch('quantmind.llm.embedding.SENTENCE_TRANSFORMERS_AVAILABLE', True):
            with patch('quantmind.llm.embedding.SentenceTransformer'):
                embedding_block = create_embedding_block(config)
                
                self.assertIsInstance(embedding_block, EmbeddingBlock)
                self.assertEqual(embedding_block.config, config)


if __name__ == "__main__":
    unittest.main()