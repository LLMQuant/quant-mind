# 🌟 Embedding Systems in QuantMind 🌟

## 📋 Table of Contents

<details>
  <summary><strong>📜 Contents</strong></summary>
  <ol>
    <li><a href="#overview">📌 Overview</a></li>
    <li><a href="#theoretical-background">📌 Theoretical Background</a></li>
    <li><a href="#architecture">📌 Architecture</a></li>
    <li><a href="#configuration">📌 Configuration</a></li>
    <li><a href="#usage-examples">📌 Usage Examples</a></li>
    <li><a href="#advanced-features">📌 Advanced Features</a></li>
    <li><a href="#best-practices">📌 Best Practices</a></li>
  </ol>
</details>

## 📌 Overview

Embeddings are numerical representations of text that capture semantic meaning in high-dimensional vector spaces. In quantitative finance, embeddings enable:

- **Document Analysis**: Converting financial reports into searchable vectors
- **Semantic Search**: Finding similar financial documents
- **Content Clustering**: Grouping related financial information
- **Feature Engineering**: Creating numerical features from textual data

QuantMind provides a flexible embedding system through the `EmbeddingBlock` class, supporting multiple providers via a unified interface.

## 📌 Theoretical Background

### What are Embeddings?

Embeddings map discrete objects (words, sentences, documents) to continuous vector spaces where:
- **Similar objects** are positioned close to each other
- **Mathematical operations** have semantic meaning
- **Dimensionality** typically ranges from 100 to 1536 dimensions

### Supported Models

| Model | Dimensions | Use Case | Provider |
|-------|------------|----------|----------|
| `text-embedding-ada-002` | 1536 | General purpose | OpenAI |
| `text-embedding-3-small` | 1536 | High performance | OpenAI |
| `text-embedding-3-large` | 3072 | Maximum quality | OpenAI |
| `embed-english-v3.0` | 1024 | Multilingual | Cohere |

## 📌 Architecture

### Core Components

```python
from quantmind.config import EmbeddingConfig
from quantmind.llm import EmbeddingBlock, create_embedding_block
```

#### EmbeddingConfig
Manages all embedding parameters:

```python
class EmbeddingConfig(BaseModel):
    model: str = "text-embedding-ada-002"
    user: Optional[str] = None
    dimensions: Optional[int] = None
    encoding_format: str = "float"
    timeout: int = 600
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    api_type: Optional[str] = None
```

#### EmbeddingBlock
Main interface for generating embeddings:

```python
class EmbeddingBlock:
    def generate_embedding(self, text: str) -> Optional[List[float]]
    def generate_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]
    def batch_embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]
    def test_connection(self) -> bool
    def get_info(self) -> Dict[str, Any]
```

## 📌 Configuration

### Basic Setup

```python
from quantmind.config import EmbeddingConfig

# Simple configuration
config = EmbeddingConfig(
    model="text-embedding-ada-002",
    api_key="your-api-key"
)
```

### Advanced Configuration

```python
# Custom dimensions (OpenAI text-embedding-3+)
config = EmbeddingConfig(
    model="text-embedding-3-small",
    dimensions=512,  # Reduce from default 1536
    encoding_format="float",
    timeout=30
)

# Azure OpenAI
config = EmbeddingConfig(
    model="text-embedding-ada-002",
    api_key="azure-key",
    api_base="https://your-resource.openai.azure.com/",
    api_version="2023-05-15",
    api_type="azure"
)
```

## 📌 Usage Examples

### Basic Embedding Generation

```python
from quantmind.config import EmbeddingConfig
from quantmind.llm import create_embedding_block

# Create configuration
config = EmbeddingConfig(
    model="text-embedding-ada-002",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Create embedding block
embedding_block = create_embedding_block(config)

# Generate single embedding
text = "Apple Inc. reported strong quarterly earnings with revenue growth of 15%."
embedding = embedding_block.generate_embedding(text)

if embedding:
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")
```

### Batch Processing

```python
# Generate multiple embeddings
texts = [
    "Apple Inc. reported strong quarterly earnings.",
    "Microsoft Corp. announced new AI initiatives.",
    "Tesla Inc. delivered record vehicle production.",
    "Amazon.com Inc. expanded cloud services."
]

embeddings = embedding_block.generate_embeddings(texts)

if embeddings:
    print(f"Generated {len(embeddings)} embeddings")
    for i, emb in enumerate(embeddings):
        print(f"Text {i+1}: {len(emb)} dimensions")
```

### Semantic Similarity

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    embeddings = embedding_block.generate_embeddings([text1, text2])
    
    if embeddings and len(embeddings) == 2:
        similarity = cosine_similarity(
            [embeddings[0]], 
            [embeddings[1]]
        )[0][0]
        return similarity
    
    return 0.0

# Example usage
text1 = "Apple's iPhone sales exceeded expectations"
text2 = "iPhone revenue surpassed analyst predictions"
similarity = calculate_similarity(text1, text2)
print(f"Similarity: {similarity:.3f}")
```

## 📌 Advanced Features

### Custom Dimensions

```python
# Reduce embedding dimensions for efficiency
config = EmbeddingConfig(
    model="text-embedding-3-small",
    dimensions=512  # Reduce from 1536 to 512
)

embedding_block = create_embedding_block(config)
embedding = embedding_block.generate_embedding("Sample text")
print(f"Reduced dimensions: {len(embedding)}")  # 512
```

### Connection Testing

```python
# Test API connection before processing
if embedding_block.test_connection():
    print("✅ API connection successful")
    # Proceed with embedding generation
else:
    print("❌ API connection failed")
    # Handle error or retry
```

### Configuration Information

```python
# Get detailed configuration information
info = embedding_block.get_info()
print(f"Model: {info['model']}")
print(f"Provider: {info['provider']}")
print(f"Dimensions: {info['dimension']}")
print(f"Format: {info['encoding_format']}")
```

## 📌 Best Practices

### 1. Model Selection

| Use Case | Recommended Model | Reasoning |
|----------|------------------|-----------|
| **General purpose** | `text-embedding-ada-002` | Good balance of quality and cost |
| **High performance** | `text-embedding-3-small` | Better quality, slightly higher cost |
| **Maximum quality** | `text-embedding-3-large` | Best quality, highest cost |
| **Multilingual** | `embed-multilingual-v3.0` | Support for multiple languages |

### 2. Batch Processing

```python
# Efficient batch processing
def efficient_batch_embedding(texts: List[str], batch_size: int = 100):
    """Process texts in optimal batches."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_embeddings = embedding_block.generate_embeddings(batch)
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
    
    return all_embeddings
```

### 3. Error Handling

```python
def robust_embedding_generation(text: str, max_retries: int = 3):
    """Generate embedding with retry logic."""
    for attempt in range(max_retries):
        try:
            embedding = embedding_block.generate_embedding(text)
            if embedding:
                return embedding
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

### 4. Caching

```python
import hashlib
import pickle
import os

class CachedEmbeddingBlock:
    def __init__(self, embedding_block, cache_dir: str = "embedding_cache"):
        self.embedding_block = embedding_block
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        # Create cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_file = os.path.join(self.cache_dir, f"{text_hash}.pkl")
        
        # Check cache
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Generate embedding
        embedding = self.embedding_block.generate_embedding(text)
        
        # Cache result
        if embedding:
            with open(cache_file, 'wb') as f:
                pickle.dump(embedding, f)
        
        return embedding
```

## 📚 Related Documentation

- [EmbeddingBlock API Reference](../docs/EMBEDDINGS.md)
- [Examples](../examples/llm/embedding_block_example.py)
- [Configuration Guide](../quantmind/config/embedding.py)

## 🔗 External Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Embeddings](https://docs.cohere.com/docs/embeddings)
- [Vector Similarity Search](https://www.pinecone.io/learn/vector-similarity-search/)

---

*Last updated: January 2025* 