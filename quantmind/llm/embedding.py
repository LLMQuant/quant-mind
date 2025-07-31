"""EmbeddingBlock - A reusable Embedding function block using LiteLLM."""

import os
import time
import numpy as np
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from quantmind.utils.logger import get_logger

from ..config import EmbeddingConfig

logger = get_logger(__name__)

try:
    import litellm
    from litellm import embedding
    
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

class EmbeddingBlock:
	"""A reusable Embedding function block using LiteLLM.
	
	EmbeddingBlock provides a consistent interface for generating embeddings across different providers (OpenAI, Anthropic, etc.).
	
	Unlike workflows, EmbeddingBlock focuses on providing basic embedding capabilities without business logic."""
	
	def __init__(self, config: EmbeddingConfig):
		"""Initialize the EmbeddingBlock with configuration.
		
		Args:
			config: Embedding configuration
		
		Raises:
			ImportError: If LiteLLM is not available.
		"""
		if not LITELLM_AVAILABLE:
			raise ImportError("litellm is required for EmbeddingBlock but not installed.")
		
		self.config = config
		self._setup_litellm()
		
		logger.info(f"Initialized EmbeddingBlock with model: {config.model}")

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
			elif provider_type == "azure":
				os.environ["AZURE_API_KEY"] = self.config.api_key
			elif provider_type == "cohere":
				os.environ["COHERE_API_KEY"] = self.config.api_key
		
		logger.debug(f"Configured LiteLLM for provider: {self.config.get_provider_type()}")
	
	
	def generate_embedding(self, text: str, **kwargs) -> Optional[List[float]]:
		"""Generate embedding using the configured Embedding model.

		Args:
			text (str): The input text to embed.
			**kwargs: Additional parameters to override config

		Returns:
			List[float]: The embedding vector as a list of floats, or None if failed.
		"""
		try:
			# Get LiteLLM parameters
			params = self.config.get_litellm_params()
			params.update(kwargs)  # Allow runtime overrides
			
			# Add input text
			params["input"] = text
			
			# Call LiteLLM embedding
			response = self._call_with_retry(params)
			
			if response and hasattr(response, 'data'):
				# Extract embedding from response
				embedding_data = response.data[0] if isinstance(response.data, list) else response.data
				return embedding_data.embedding
			
			return None
			
		except Exception as e:
			logger.error(f"Failed to generate embedding: {e}")
			return None

	def generate_embeddings(self, texts: List[str], **kwargs) -> Optional[List[List[float]]]:
		"""Generate embeddings for multiple texts.

		Args:
			texts (List[str]): List of input texts to embed.
			**kwargs: Additional parameters to override config

		Returns:
			List[List[float]]: List of embedding vectors, or None if failed.
		"""
		try:
			# Get LiteLLM parameters
			params = self.config.get_litellm_params()
			params.update(kwargs)  # Allow runtime overrides
			
			# Add input texts
			params["input"] = texts
			
			# Call LiteLLM embedding
			response = self._call_with_retry(params)
			
			if response and hasattr(response, 'data'):
				# Extract embeddings from response
				return [item.embedding for item in response.data]
			
			return None
			
		except Exception as e:
			logger.error(f"Failed to generate embeddings: {e}")
			return None
	
	def _call_with_retry(self, params: Dict[str, Any]) -> Optional[Any]:
		"""Call LiteLLM embedding with retry logic.

		Args:
			params (Dict[str, Any]): The parameters to pass to the embedding function.

		Returns:
			Optional[Any]: The embedding result or None if failed.
		"""
		last_exception = None
		for attempt in range(self.config.retry_attempts + 1):
			try:
				logger.debug(
					f"Embedding call attempt {attempt + 1}/{self.config.retry_attempts + 1}"
				)
				
				# Extract input from params and remove it for the embedding call
				input_text = params.pop("input")
				response = embedding(model=self.config.model, input=input_text, **params)

				if hasattr(response, "usage") and response.usage:
					logger.debug(f"Token usage: {response.usage}")
				return response
			except Exception as e:
				last_exception = e
				logger.warning(f"Embedding call attempt {attempt + 1} failed: {e}")

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


	def test_connection(self) -> bool:
		"""Test if the embedding connection is working.

        Returns:
            True if connection is working, False otherwise
        """
		try:
			response = self.generate_embedding(
				"Hello, this is a test for embedding generation. Please respond with 'OK'."
			)
			return response is not None and len(response) > 0
		except Exception as e:
			logger.error(f"Connection test failed: {e}")
			return False


	def get_info(self) -> Dict[str, Any]:
		"""Get information about the embedding block.

		Returns:
			Dictionary with embedding block information
		"""
		info = {
			"model": self.config.model,
			"provider": self.config.get_provider_type(),
			"dimension": self.get_embedding_dimension(),
			"config": self.config.model_dump(),
		}
		return info

	def get_embedding_dimension(self) -> Optional[int]:
		"""Get the dimension of embeddings generated by this model.

		Returns:
			Embedding dimension or None if not available
		"""
		# First check if dimensions is specified in config
		if self.config.dimensions:
			return self.config.dimensions
		
		try:
			# Try to get dimension by generating a test embedding
			test_embedding = self.generate_embedding("test")
			return len(test_embedding) if test_embedding else None
		except Exception as e:
			logger.error(f"Failed to get embedding dimension: {e}")
			return None

	def update_config(self, **kwargs) -> None:
		"""Update the embedding configuration.

		Args:
			**kwargs: Configuration parameters to update
		"""
		for key, value in kwargs.items():
			if hasattr(self.config, key):
				setattr(self.config, key, value)

		logger.info(f"Updated embedding configuration: {kwargs}")

	@contextmanager
	def temporary_config(self, **kwargs):
		"""Temporarily modify configuration for a context.

		Args:
			**kwargs: Temporary configuration parameters

		Yields:
			Self with temporary configuration
		"""
		original_config = {}
		for key, value in kwargs.items():
			if hasattr(self.config, key):
				original_config[key] = getattr(self.config, key)
				setattr(self.config, key, value)

		try:
			yield self
		finally:
			# Restore original configuration
			for key, value in original_config.items():
				setattr(self.config, key, value)

	def batch_embed(
		self, 
		texts: List[str], 
		batch_size: int = 32,
		**kwargs
	) -> Optional[List[List[float]]]:
		"""Generate embeddings in batches for large datasets.

		Args:
			texts: List of texts to embed
			batch_size: Number of texts to process in each batch
			**kwargs: Additional parameters for embedding generation

		Returns:
			List of embedding vectors or None if failed
		"""
		try:
			all_embeddings = []
			
			for i in range(0, len(texts), batch_size):
				batch = texts[i:i + batch_size]
				batch_embeddings = self.generate_embeddings(batch, **kwargs)
				
				if batch_embeddings is None:
					logger.error(f"Failed to generate embeddings for batch {i//batch_size}")
					return None
				
				all_embeddings.extend(batch_embeddings)
				
				# Add delay between batches if specified
				if self.config.retry_delay > 0 and i + batch_size < len(texts):
					time.sleep(self.config.retry_delay)

			return all_embeddings

		except Exception as e:
			logger.error(f"Batch embedding failed: {e}")
			return None


def create_embedding_block(config: EmbeddingConfig) -> EmbeddingBlock:
	"""Create an EmbeddingBlock instance.

	Args:
		config: Embedding configuration

	Returns:
		Configured EmbeddingBlock instance
	"""
	return EmbeddingBlock(config)