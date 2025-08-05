# EmbeddingBlock Documentation

## Overview

The `EmbeddingBlock` class provides a flexible interface for generating text embeddings using LiteLLM. It supports multiple embedding providers through a single, consistent API.

## Quick Start

```python
from quantmind.config import EmbeddingConfig
from quantmind.llm import create_embedding_block

# Simple configuration
config = EmbeddingConfig(
    model="text-embedding-ada-002"
)

embedding_block = create_embedding_block(config)
embedding = embedding_block.generate_embedding("Sample text")
```

## Configuration

### Required Parameters
- `model`: Embedding model name (e.g., "text-embedding-ada-002")

### Optional Parameters
- `user`: Unique identifier for end-user
- `dimensions`: Number of dimensions (OpenAI text-embedding-3+)
- `encoding_format`: "float" or "base64" (default: "float")
- `timeout`: Request timeout in seconds (default: 600)
- `api_base`: Custom API endpoint
- `api_version`: Azure-specific API version
- `api_key`: API key for authentication
- `api_type`: Type of API to use

## Examples

### Basic Usage
```python
config = EmbeddingConfig(model="text-embedding-ada-002")
embedding_block = create_embedding_block(config)
embedding = embedding_block.generate_embedding("Text to embed")
```

### With Custom Dimensions
```python
config = EmbeddingConfig(
    model="text-embedding-3-small",
    dimensions=512
)
```

### Azure OpenAI
```python
config = EmbeddingConfig(
    model="text-embedding-ada-002",
    api_key="azure-key",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2023-05-15",
    api_type="azure"
)
```

## Methods

- `generate_embedding(text)`: Generate single embedding
- `generate_embeddings(texts)`: Generate multiple embeddings
- `batch_embed(texts, batch_size)`: Process large datasets
- `test_connection()`: Test API connection
- `get_info()`: Get configuration information
- `get_embedding_dimension()`: Get embedding dimension

## See Also

- `examples/llm/embedding_block_example.py` for complete examples
