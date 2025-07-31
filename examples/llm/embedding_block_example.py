"""Example usage of EmbeddingBlock for different embedding providers."""

import os
from typing import List

from quantmind.config import EmbeddingConfig
from quantmind.llm import EmbeddingBlock, create_embedding_block
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


def example_openai_embeddings():
    """Example using OpenAI embeddings."""
    print("\n=== OpenAI Embeddings Example ===")

    # Configuration for OpenAI embeddings
    config = EmbeddingConfig(
        model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY"),
        timeout=30,
        encoding_format="float",
    )

    # Create embedding block
    embedding_block = create_embedding_block(config)

    # Test connection
    if embedding_block.test_connection():
        print("✅ OpenAI connection successful")
    else:
        print("❌ OpenAI connection failed")
        return

    # Generate single embedding
    text = "This is a sample text for embedding generation."
    embedding = embedding_block.generate_embedding(text)

    if embedding:
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")

    # Generate multiple embeddings
    texts = [
        "First sample text for embedding.",
        "Second sample text with different content.",
        "Third sample text for batch processing.",
    ]

    embeddings = embedding_block.generate_embeddings(texts)

    if embeddings:
        print(f"✅ Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"   Text {i + 1}: {len(emb)} dimensions")

    # Get embedding information
    info = embedding_block.get_info()
    print(f"📊 Model info: {info['model']}")
    print(f"📊 Provider: {info['provider']}")
    print(f"📊 Dimension: {info['dimension']}")


def example_openai_text_embedding_3():
    """Example using OpenAI text-embedding-3 with custom dimensions."""
    print("\n=== OpenAI Text-Embedding-3 Example ===")

    # Configuration for OpenAI text-embedding-3
    config = EmbeddingConfig(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        dimensions=512,  # Custom dimension (default is 1536)
        timeout=30,
        encoding_format="float",
    )

    # Create embedding block
    embedding_block = create_embedding_block(config)

    # Test connection
    if embedding_block.test_connection():
        print("✅ OpenAI text-embedding-3 connection successful")
    else:
        print("❌ OpenAI text-embedding-3 connection failed")
        return

    # Generate single embedding
    text = (
        "This is a sample text for embedding generation with custom dimensions."
    )
    embedding = embedding_block.generate_embedding(text)

    if embedding:
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")

    # Generate multiple embeddings
    texts = [
        "First sample text for embedding.",
        "Second sample text with different content.",
        "Third sample text for batch processing.",
    ]

    embeddings = embedding_block.generate_embeddings(texts)

    if embeddings:
        print(f"✅ Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"   Text {i + 1}: {len(emb)} dimensions")

    # Get embedding information
    info = embedding_block.get_info()
    print(f"📊 Model info: {info['model']}")
    print(f"📊 Provider: {info['provider']}")
    print(f"📊 Dimension: {info['dimension']}")


def example_azure_embeddings():
    """Example using Azure OpenAI embeddings."""
    print("\n=== Azure OpenAI Embeddings Example ===")

    # Configuration for Azure OpenAI embeddings
    config = EmbeddingConfig(
        model="text-embedding-ada-002",
        api_key=os.getenv("AZURE_API_KEY"),
        api_base=os.getenv("AZURE_API_BASE"),
        api_version=os.getenv("AZURE_API_VERSION", "2023-05-15"),
        api_type="azure",
        timeout=30,
        encoding_format="float",
    )

    # Create embedding block
    embedding_block = create_embedding_block(config)

    # Create embedding block
    embedding_block = create_embedding_block(config)

    # Test connection
    if embedding_block.test_connection():
        print("✅ Azure OpenAI connection successful")
    else:
        print("❌ Azure OpenAI connection failed")
        return

    # Generate single embedding
    text = "This is a sample text for Azure OpenAI embedding generation."
    embedding = embedding_block.generate_embedding(text)

    if embedding:
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        print(f"   First 5 values: {embedding[:5]}")

    # Generate multiple embeddings
    texts = [
        "First sample text for Azure embedding.",
        "Second sample text with different content.",
        "Third sample text for batch processing.",
    ]

    embeddings = embedding_block.generate_embeddings(texts)

    if embeddings:
        print(f"✅ Generated {len(embeddings)} embeddings")
        for i, emb in enumerate(embeddings):
            print(f"   Text {i + 1}: {len(emb)} dimensions")

    # Get embedding information
    info = embedding_block.get_info()
    print(f"📊 Model info: {info['model']}")
    print(f"📊 Provider: {info['provider']}")
    print(f"📊 Dimension: {info['dimension']}")


def example_configuration_variants():
    """Example showing different configuration variants."""
    print("\n=== Configuration Variants Example ===")

    # Base configuration
    base_config = EmbeddingConfig(
        model="text-embedding-ada-002",
        api_key=os.getenv("OPENAI_API_KEY"),
        encoding_format="float",
    )

    # Create variants with different parameters
    fast_config = base_config.create_variant(timeout=10, retry_attempts=1)

    conservative_config = base_config.create_variant(
        timeout=120, retry_attempts=5, retry_delay=2.0
    )

    print(f"Base config timeout: {base_config.timeout}s")
    print(f"Fast config timeout: {fast_config.timeout}s")
    print(f"Conservative config timeout: {conservative_config.timeout}s")

    # Test with temporary configuration
    embedding_block = create_embedding_block(base_config)

    with embedding_block.temporary_config(timeout=5):
        print("Using temporary configuration with 5s timeout")
        # Any embedding operations here will use the temporary config
        embedding = embedding_block.generate_embedding("Test with temp config")
        if embedding:
            print("✅ Temporary configuration worked")


def example_error_handling():
    """Example showing error handling and fallbacks."""
    print("\n=== Error Handling Example ===")

    # Try with invalid API key
    config = EmbeddingConfig(
        model="text-embedding-ada-002",
        api_key="invalid_key",
        timeout=5,
    )

    embedding_block = create_embedding_block(config)

    # This should fail gracefully
    embedding = embedding_block.generate_embedding("Test text")
    if embedding is None:
        print("✅ Gracefully handled invalid API key")

    # Try with non-existent model
    config = EmbeddingConfig(
        model="non-existent-model",
        timeout=5,
    )

    try:
        embedding_block = create_embedding_block(config)
        print("❌ Should have failed with non-existent model")
    except Exception as e:
        print(f"✅ Gracefully handled non-existent model: {e}")


def main():
    """Run all embedding examples."""
    print("🚀 EmbeddingBlock Examples")
    print("=" * 50)

    # Run examples based on available API keys
    if os.getenv("OPENAI_API_KEY"):
        example_openai_embeddings()
        example_openai_text_embedding_3()
    else:
        print("\n⚠️  Skipping OpenAI examples - OPENAI_API_KEY not set")

    if os.getenv("AZURE_API_KEY") and os.getenv("AZURE_API_BASE"):
        example_azure_embeddings()
    else:
        print(
            "\n⚠️  Skipping Azure example - AZURE_API_KEY or AZURE_API_BASE not set"
        )

    example_configuration_variants()
    example_error_handling()

    print("\n✅ All examples completed!")


if __name__ == "__main__":
    main()
