"""LLM module for QuantMind - Basic LLM functionality."""

from .block import LLMBlock, create_llm_block
from .embedding import EmbeddingBlock, create_embedding_block

__all__ = [
    "LLMBlock",
    "create_llm_block"
]
