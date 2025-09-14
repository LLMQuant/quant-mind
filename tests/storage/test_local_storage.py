"""Tests for enhanced storage functionality with efficient indexing."""

import json
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from quantmind.config.storage import LocalStorageConfig
from quantmind.models.paper import Paper
from quantmind.storage.local_storage import LocalStorage


class TestEnhancedStorageWithIndexing(unittest.TestCase):
    """Test enhanced storage functionality with efficient indexing."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = LocalStorageConfig(
            storage_dir=self.temp_dir, download_timeout=1
        )
        self.storage = LocalStorage(self.config)

    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_index_initialization(self):
        """Test that indexes are properly initialized."""
        # Check that index files are created
        self.assertTrue(self.storage._get_index_path("raw_files").exists())
        self.assertTrue(self.storage._get_index_path("knowledges").exists())
        self.assertTrue(self.storage._get_index_path("embeddings").exists())

        # Check that indexes are empty initially
        self.assertEqual(len(self.storage._raw_files_index), 0)
        self.assertEqual(len(self.storage._knowledges_index), 0)
        self.assertEqual(len(self.storage._embeddings_index), 0)

    def test_raw_file_indexing(self):
        """Test raw file storage and indexing."""
        # Store a raw file
        pdf_content = b"%PDF-1.4 test content"
        file_path = self.storage.store_raw_file(
            file_id="test_pdf", content=pdf_content, file_extension=".pdf"
        )

        # Check that index was updated
        self.assertIn("test_pdf", self.storage._raw_files_index)
        index_entry = self.storage._raw_files_index["test_pdf"]
        self.assertEqual(index_entry["extension"], ".pdf")

        # Check that index file was saved
        index_path = self.storage._get_index_path("raw_files")
        self.assertTrue(index_path.exists())

        with open(index_path, "r") as f:
            saved_index = json.load(f)
        self.assertIn("test_pdf", saved_index)

    def test_fast_raw_file_lookup(self):
        """Test that raw file lookup uses index for fast retrieval."""
        # Store multiple files
        for i in range(5):
            content = f"test content {i}".encode()
            self.storage.store_raw_file(
                file_id=f"test_{i}", content=content, file_extension=".txt"
            )

        # Verify all files are in index
        self.assertEqual(len(self.storage._raw_files_index), 5)

        # Test retrieval - should use index
        retrieved_path = self.storage.get_raw_file("test_3")
        self.assertIsNotNone(retrieved_path)
        self.assertTrue(retrieved_path.exists())
        self.assertEqual(retrieved_path.suffix, ".txt")

    def test_knowledge_indexing(self):
        """Test knowledge item storage and indexing."""
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract",
            authors=["Test Author"],
            arxiv_id="test.001",
            categories=["q-fin.CP"],
            published_date=datetime.now(timezone.utc),
            source="test",
        )

        # Store knowledge
        paper_id = self.storage.store_knowledge(paper)

        # Check that index was updated
        self.assertIn("test.001", self.storage._knowledges_index)

        # Check fast retrieval
        retrieved_paper = self.storage.get_knowledge("test.001")
        self.assertIsNotNone(retrieved_paper)
        self.assertEqual(retrieved_paper.title, "Test Paper")

    def test_embedding_indexing(self):
        """Test embedding storage and indexing."""
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Store embedding
        self.storage.store_embedding("test_knowledge", embedding, "test_model")

        # Check that index was updated
        self.assertIn("test_knowledge", self.storage._embeddings_index)

        # Check fast retrieval
        retrieved_embedding = self.storage.get_embedding("test_knowledge")
        self.assertIsNotNone(retrieved_embedding)
        self.assertEqual(retrieved_embedding["embedding"], embedding)
        self.assertEqual(retrieved_embedding["model"], "test_model")

    def test_index_persistence_across_restarts(self):
        """Test that indexes persist across storage restarts."""
        # Store some data
        pdf_content = b"test pdf"
        self.storage.store_raw_file(
            "test_pdf", content=pdf_content, file_extension=".pdf"
        )

        paper = Paper(
            title="Test Paper",
            abstract="Test abstract",
            authors=["Test Author"],
            arxiv_id="test.001",
            source="test",
        )
        self.storage.store_knowledge(paper)

        # Create new storage instance (simulating restart)
        new_storage = LocalStorage(self.config)

        # Check that indexes were loaded
        self.assertIn("test_pdf", new_storage._raw_files_index)
        self.assertIn("test.001", new_storage._knowledges_index)

        # Check that retrieval still works
        retrieved_pdf = new_storage.get_raw_file("test_pdf")
        self.assertIsNotNone(retrieved_pdf)

        retrieved_paper = new_storage.get_knowledge("test.001")
        self.assertIsNotNone(retrieved_paper)

    def test_index_rebuilding(self):
        """Test index rebuilding functionality."""
        # Create files directly in filesystem (bypassing storage)
        raw_file = self.config.raw_files_dir / "direct_file.pdf"
        raw_file.write_bytes(b"direct pdf content")

        knowledge_file = self.config.knowledges_dir / "direct_knowledge.json"
        knowledge_data = {
            "id": "direct_knowledge",
            "title": "Direct Knowledge",
            "abstract": "Direct abstract",
            "content_type": "generic",
            "source": "direct",
        }
        knowledge_file.write_text(json.dumps(knowledge_data))

        # Rebuild indexes
        self.storage.rebuild_all_indexes()

        # Check that files were indexed
        self.assertIn("direct_file", self.storage._raw_files_index)
        self.assertIn("direct_knowledge", self.storage._knowledges_index)

        # Check retrieval works
        retrieved_raw = self.storage.get_raw_file("direct_file")
        self.assertIsNotNone(retrieved_raw)

        retrieved_knowledge = self.storage.get_knowledge("direct_knowledge")
        self.assertIsNotNone(retrieved_knowledge)

    def test_index_cleanup_on_missing_files(self):
        """Test that index is cleaned up when files are deleted externally."""
        # Store a file
        self.storage.store_raw_file(
            "test_file", content=b"test", file_extension=".txt"
        )

        # Verify it's in index
        self.assertIn("test_file", self.storage._raw_files_index)

        # Delete file directly from filesystem
        file_path = self.storage.get_raw_file("test_file")
        file_path.unlink()

        # Try to retrieve - should clean up index
        retrieved = self.storage.get_raw_file("test_file")
        self.assertIsNone(retrieved)

        # Check that index was cleaned up
        self.assertNotIn("test_file", self.storage._raw_files_index)

    def test_fallback_to_directory_scan(self):
        """Test fallback to directory scan when file not in index."""
        # Create file directly in filesystem
        raw_file = self.config.raw_files_dir / "fallback_test.pdf"
        raw_file.write_bytes(b"fallback content")

        # File should not be in index initially
        self.assertNotIn("fallback_test", self.storage._raw_files_index)

        # Try to retrieve - should find via directory scan and add to index
        retrieved = self.storage.get_raw_file("fallback_test")
        self.assertIsNotNone(retrieved)

        # Check that it was added to index
        self.assertIn("fallback_test", self.storage._raw_files_index)

    def test_delete_operations_update_index(self):
        """Test that delete operations properly update indexes."""
        # Store and then delete raw file
        self.storage.store_raw_file(
            "delete_test", content=b"test", file_extension=".txt"
        )
        self.assertIn("delete_test", self.storage._raw_files_index)

        deleted = self.storage.delete_raw_file("delete_test")
        self.assertTrue(deleted)
        self.assertNotIn("delete_test", self.storage._raw_files_index)

        # Store and then delete knowledge
        paper = Paper(
            title="Delete Test",
            abstract="Test",
            arxiv_id="delete.001",
            source="test",
        )
        self.storage.store_knowledge(paper)
        self.assertIn("delete.001", self.storage._knowledges_index)

        deleted = self.storage.delete_knowledge("delete.001")
        self.assertTrue(deleted)
        self.assertNotIn("delete.001", self.storage._knowledges_index)

    def test_get_all_knowledges_uses_index(self):
        """Test that get_all_knowledges uses index for efficient iteration."""
        # Store multiple knowledge items
        for i in range(3):
            paper = Paper(
                title=f"Paper {i}",
                abstract=f"Abstract {i}",
                arxiv_id=f"test.{i:03d}",
                source="test",
            )
            self.storage.store_knowledge(paper)

        # Get all knowledges
        all_knowledges = list(self.storage.get_all_knowledges())

        # Should have 3 items
        self.assertEqual(len(all_knowledges), 3)

        # Check that we got the right items
        titles = [k.title for k in all_knowledges]
        self.assertIn("Paper 0", titles)
        self.assertIn("Paper 1", titles)
        self.assertIn("Paper 2", titles)

    def test_storage_info_includes_index_stats(self):
        """Test that storage info includes index statistics."""
        # Store some test data
        self.storage.store_raw_file(
            "test_file", content=b"test", file_extension=".txt"
        )

        paper = Paper(
            title="Test Paper",
            abstract="Test",
            arxiv_id="test.001",
            source="test",
        )
        self.storage.store_knowledge(paper)

        self.storage.store_embedding("test.001", [0.1, 0.2], "test_model")

        # Get storage info
        info = self.storage.get_storage_info()

        # Check that index stats are included
        self.assertIn("indexes", info)
        indexes = info["indexes"]

        self.assertEqual(indexes["raw_files"]["entries"], 1)
        self.assertEqual(indexes["knowledges"]["entries"], 1)
        self.assertEqual(indexes["embeddings"]["entries"], 1)

        # Check that index file paths are included
        self.assertIn("index_file", indexes["raw_files"])
        self.assertIn("index_file", indexes["knowledges"])
        self.assertIn("index_file", indexes["embeddings"])

    def test_process_knowledge_paper(self):
        """Test specialized Paper storage with indexing."""
        paper = Paper(
            title="Test Paper",
            abstract="Test abstract for paper",
            authors=["Test Author"],
            arxiv_id="test.001",
            categories=["q-fin.CP"],
            published_date=datetime.now(timezone.utc),
            source="test",
        )

        # Store using specialized method
        paper_id = self.storage.process_knowledge(paper)

        # Verify paper was stored and indexed
        self.assertEqual(paper_id, "test.001")
        self.assertIn("test.001", self.storage._knowledges_index)

        # Verify paper can be retrieved quickly
        retrieved_paper = self.storage.get_knowledge(paper_id)
        self.assertIsNotNone(retrieved_paper)
        self.assertEqual(retrieved_paper.title, "Test Paper")

    @patch("requests.get")
    def test_process_knowledge_paper_with_pdf_url(self, mock_requests):
        """Test Paper storage with PDF URL and indexing."""
        # Mock requests to avoid real network calls
        mock_response = Mock()
        mock_response.content = b"%PDF-1.4 fake content"
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response

        paper = Paper(
            title="Paper with PDF URL",
            abstract="Test paper with PDF URL",
            authors=["Test Author"],
            arxiv_id="test.002",
            pdf_url="https://example.com/paper.pdf",
            categories=["q-fin.CP"],
            published_date=datetime.now(timezone.utc),
            source="test",
        )

        # Store using specialized method
        paper_id = self.storage.process_knowledge(paper)

        # Verify paper was stored and indexed
        self.assertEqual(paper_id, "test.002")
        self.assertIn("test.002", self.storage._knowledges_index)

        retrieved_paper = self.storage.get_knowledge(paper_id)
        self.assertIsNotNone(retrieved_paper)
        self.assertEqual(
            retrieved_paper.pdf_url, "https://example.com/paper.pdf"
        )

    def test_store_raw_file_with_content(self):
        """Test storing raw file from content bytes with indexing."""
        # Test PDF content
        pdf_content = b"%PDF-1.4 test content"

        file_path = self.storage.store_raw_file(
            file_id="test_pdf", content=pdf_content, file_extension=".pdf"
        )

        # Verify file was created and indexed
        stored_path = Path(file_path)
        self.assertTrue(stored_path.exists())
        self.assertEqual(stored_path.suffix, ".pdf")
        self.assertIn("test_pdf", self.storage._raw_files_index)

        # Verify content
        with open(stored_path, "rb") as f:
            self.assertEqual(f.read(), pdf_content)

    def test_store_raw_file_validation(self):
        """Test input validation for store_raw_file."""
        # Test missing both parameters
        with self.assertRaises(ValueError):
            self.storage.store_raw_file("test_id")

        # Test providing both parameters
        with self.assertRaises(ValueError):
            self.storage.store_raw_file(
                "test_id", file_path=Path("dummy"), content=b"dummy"
            )

    def test_store_raw_file_backward_compatibility(self):
        """Test that file copying still works with indexing."""
        # Create a temporary file to copy
        temp_file = self.temp_dir / "source.txt"
        temp_file.write_text("Source file content")

        # Store by copying
        file_path = self.storage.store_raw_file(
            file_id="copied_file", file_path=temp_file
        )

        # Verify file was copied and indexed
        stored_path = Path(file_path)
        self.assertTrue(stored_path.exists())
        self.assertEqual(stored_path.read_text(), "Source file content")
        self.assertIn("copied_file", self.storage._raw_files_index)


if __name__ == "__main__":
    unittest.main()
