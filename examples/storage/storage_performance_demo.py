"""Storage performance demonstration with and without indexing.

This script demonstrates the dramatic performance improvement achieved
by the new indexing system in LocalStorage.
"""

import time
from pathlib import Path
from datetime import datetime, timezone

from quantmind.config.storage import LocalStorageConfig
from quantmind.storage.local_storage import LocalStorage
from quantmind.models.paper import Paper


def create_test_data(storage: LocalStorage, num_items: int = 100):
    """Create test data for performance testing."""
    print(f"Creating {num_items} test items...")

    for i in range(num_items):
        # Create test papers
        paper = Paper(
            title=f"Test Paper {i}",
            abstract=f"This is test paper number {i} for performance testing.",
            authors=[f"Author {i}"],
            arxiv_id=f"test.{i:04d}",
            categories=["q-fin.CP"],
            published_date=datetime.now(timezone.utc),
            source="test",
        )
        storage.store_knowledge(paper)

        # Create test raw files
        content = f"Test content for file {i}".encode()
        storage.store_raw_file(
            f"test_file_{i}", content=content, file_extension=".txt"
        )

        # Create test embeddings
        embedding = [float(j) for j in range(10)]  # Simple embedding
        storage.store_embedding(f"test.{i:04d}", embedding, "test_model")

    print(f"✅ Created {num_items} items successfully")


def simulate_old_behavior(storage: LocalStorage, num_lookups: int = 50):
    """Simulate old file system scanning behavior for comparison."""
    print(f"\n🐌 Simulating old behavior (directory scanning)...")

    start_time = time.time()

    for i in range(num_lookups):
        # Simulate directory scanning by manually checking files
        file_id = f"test_file_{i}"
        found = False

        # This simulates the old glob-based search
        for file_path in storage.config.raw_files_dir.glob(f"{file_id}.*"):
            if file_path.is_file():
                found = True
                break

    end_time = time.time()
    old_time = end_time - start_time

    print(f"   Time for {num_lookups} lookups: {old_time:.4f} seconds")
    print(f"   Average per lookup: {(old_time/num_lookups)*1000:.2f} ms")

    return old_time


def test_new_indexing_performance(storage: LocalStorage, num_lookups: int = 50):
    """Test the new indexing system performance."""
    print(f"\n🚀 Testing new indexing system...")

    start_time = time.time()

    for i in range(num_lookups):
        # Use the new index-based lookup
        file_id = f"test_file_{i}"
        file_path = storage.get_raw_file(file_id)
        # Just verify it exists
        assert file_path is not None

    end_time = time.time()
    new_time = end_time - start_time

    print(f"   Time for {num_lookups} lookups: {new_time:.4f} seconds")
    print(f"   Average per lookup: {(new_time/num_lookups)*1000:.2f} ms")

    return new_time


def test_knowledge_lookup_performance(
    storage: LocalStorage, num_lookups: int = 50
):
    """Test knowledge lookup performance."""
    print(f"\n📚 Testing knowledge lookup performance...")

    start_time = time.time()

    for i in range(num_lookups):
        knowledge_id = f"test.{i:04d}"
        knowledge = storage.get_knowledge(knowledge_id)
        assert knowledge is not None

    end_time = time.time()
    lookup_time = end_time - start_time

    print(
        f"   Time for {num_lookups} knowledge lookups: {lookup_time:.4f} seconds"
    )
    print(f"   Average per lookup: {(lookup_time/num_lookups)*1000:.2f} ms")

    return lookup_time


def test_batch_operations(storage: LocalStorage):
    """Test batch operations performance."""
    print(f"\n📦 Testing batch operations...")

    # Test get_all_knowledges (now uses index)
    start_time = time.time()
    all_knowledges = list(storage.get_all_knowledges())
    end_time = time.time()

    batch_time = end_time - start_time
    count = len(all_knowledges)

    print(f"   Retrieved {count} knowledge items in {batch_time:.4f} seconds")
    print(f"   Average per item: {(batch_time/count)*1000:.2f} ms")

    return batch_time


def test_index_rebuild_performance(storage: LocalStorage):
    """Test index rebuilding performance."""
    print(f"\n🔄 Testing index rebuild performance...")

    start_time = time.time()
    storage.rebuild_all_indexes()
    end_time = time.time()

    rebuild_time = end_time - start_time

    print(f"   Index rebuild time: {rebuild_time:.4f} seconds")
    print(f"   Raw files indexed: {len(storage._raw_files_index)}")
    print(f"   Knowledge items indexed: {len(storage._knowledges_index)}")
    print(f"   Embeddings indexed: {len(storage._embeddings_index)}")

    return rebuild_time


def show_storage_statistics(storage: LocalStorage):
    """Show detailed storage statistics."""
    print(f"\n📊 Storage Statistics:")

    info = storage.get_storage_info()

    print(f"   Storage Directory: {info['storage_dir']}")
    print(f"   Knowledge Count: {info['knowledge_count']}")
    print(f"   Raw Files Count: {info['raw_files_count']}")
    print(f"   Embeddings Count: {info['embeddings_count']}")

    print(f"\n   Index Statistics:")
    for index_type, stats in info["indexes"].items():
        print(f"     {index_type}: {stats['entries']} entries")
        print(f"       Index file: {Path(stats['index_file']).name}")


def main():
    """Main performance demonstration."""
    print("🎯 QuantMind Storage Performance Demonstration")
    print("=" * 60)

    # Setup
    demo_dir = Path("./performance_demo_data")
    if demo_dir.exists():
        import shutil

        shutil.rmtree(demo_dir)

    config = LocalStorageConfig(storage_dir=demo_dir)
    storage = LocalStorage(config)

    try:
        # Create test data
        num_items = 100
        num_lookups = 50

        create_test_data(storage, num_items)

        # Performance tests
        old_time = simulate_old_behavior(storage, num_lookups)
        new_time = test_new_indexing_performance(storage, num_lookups)
        knowledge_time = test_knowledge_lookup_performance(storage, num_lookups)
        batch_time = test_batch_operations(storage)
        rebuild_time = test_index_rebuild_performance(storage)

        # Calculate improvement
        if old_time > 0:
            speedup = old_time / new_time
            improvement = ((old_time - new_time) / old_time) * 100
        else:
            speedup = float("inf")
            improvement = 100

        # Summary
        print(f"\n" + "=" * 60)
        print(f"🎉 PERFORMANCE SUMMARY")
        print(f"=" * 60)
        print(f"   Test Items: {num_items}")
        print(f"   Lookups Tested: {num_lookups}")
        print(f"")
        print(f"   Old Method (directory scan): {old_time:.4f}s")
        print(f"   New Method (index lookup):   {new_time:.4f}s")
        print(f"")
        print(f"   🚀 Speedup: {speedup:.1f}x faster")
        print(f"   📈 Improvement: {improvement:.1f}% faster")
        print(f"")
        print(f"   📚 Knowledge lookup time: {knowledge_time:.4f}s")
        print(f"   📦 Batch retrieval time: {batch_time:.4f}s")
        print(f"   🔄 Index rebuild time: {rebuild_time:.4f}s")

        # Show storage statistics
        show_storage_statistics(storage)

        print(f"\n✨ Key Benefits:")
        print(f"   • O(1) lookup time vs O(n) directory scanning")
        print(f"   • Persistent indexes survive restarts")
        print(f"   • Automatic index rebuilding for data recovery")
        print(f"   • Self-healing: removes stale entries automatically")
        print(f"   • Fallback to directory scan for missing entries")

        print(f"\n📁 Demo data saved in: {demo_dir}")
        print(f"   Check the index files in: {demo_dir}/extra/")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        raise


if __name__ == "__main__":
    main()
