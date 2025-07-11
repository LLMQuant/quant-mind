"""Tests for storage configuration models."""

import shutil
import tempfile
import unittest
from pathlib import Path

from quantmind.config.storage import LocalStorageConfig


class TestLocalStorageConfig(unittest.TestCase):
    """Test LocalStorageConfig functionality."""

    def setUp(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.addCleanup(self._cleanup_temp_dir)

    def _cleanup_temp_dir(self):
        """Clean up temporary directory after test."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_default_configuration(self):
        """Test default configuration values."""
        config = LocalStorageConfig()

        # Check default storage directory
        self.assertEqual(
            config.storage_dir, Path("./data").expanduser().resolve()
        )

        # Check that directory was created
        self.assertTrue(config.storage_dir.exists())

        # Clean up default directory
        if config.storage_dir.exists() and config.storage_dir.name == "data":
            shutil.rmtree(config.storage_dir)

    def test_custom_storage_directory(self):
        """Test custom storage directory configuration."""
        custom_dir = self.temp_dir / "custom_storage"

        config = LocalStorageConfig(storage_dir=custom_dir)

        # Check that custom directory is set and resolved
        self.assertEqual(config.storage_dir, custom_dir.resolve())
        self.assertTrue(config.storage_dir.exists())

    def test_model_post_init_creates_directories(self):
        """Test that model_post_init creates all required subdirectories."""
        storage_dir = self.temp_dir / "test_storage"

        config = LocalStorageConfig(storage_dir=storage_dir)

        # Check that main directory exists
        self.assertTrue(storage_dir.exists())

        # Check that all subdirectories were created
        self.assertTrue((storage_dir / "raw_files").exists())
        self.assertTrue((storage_dir / "knowledges").exists())
        self.assertTrue((storage_dir / "embeddings").exists())
        self.assertTrue((storage_dir / "extra").exists())

    def test_directory_properties(self):
        """Test directory property methods."""
        storage_dir = self.temp_dir / "property_test"

        config = LocalStorageConfig(storage_dir=storage_dir)

        # Test raw_files_dir property - use resolve() for consistent path comparison
        expected_raw_files = (storage_dir / "raw_files").resolve()
        self.assertEqual(config.raw_files_dir.resolve(), expected_raw_files)
        self.assertTrue(config.raw_files_dir.exists())

        # Test knowledges_dir property
        expected_knowledges = (storage_dir / "knowledges").resolve()
        self.assertEqual(config.knowledges_dir.resolve(), expected_knowledges)
        self.assertTrue(config.knowledges_dir.exists())

        # Test embeddings_dir property
        expected_embeddings = (storage_dir / "embeddings").resolve()
        self.assertEqual(config.embeddings_dir.resolve(), expected_embeddings)
        self.assertTrue(config.embeddings_dir.exists())

        # Test extra_dir property
        expected_extra = (storage_dir / "extra").resolve()
        self.assertEqual(config.extra_dir.resolve(), expected_extra)
        self.assertTrue(config.extra_dir.exists())

    def test_path_expansion_and_resolution(self):
        """Test that paths are properly expanded and resolved."""
        # Test with relative path
        relative_path = Path("./relative_storage")
        config = LocalStorageConfig(storage_dir=relative_path)

        # Should be converted to absolute path
        self.assertTrue(config.storage_dir.is_absolute())
        self.assertEqual(config.storage_dir.name, "relative_storage")

        # Clean up
        if config.storage_dir.exists():
            shutil.rmtree(config.storage_dir)

    def test_home_directory_expansion(self):
        """Test that ~ in paths is properly expanded."""
        # Test with home directory path
        home_path = Path("~/test_quantmind_storage")
        config = LocalStorageConfig(storage_dir=home_path)

        # Should expand ~ to actual home directory
        self.assertFalse(str(config.storage_dir).startswith("~"))
        self.assertTrue(config.storage_dir.is_absolute())
        self.assertTrue(
            str(config.storage_dir).endswith("test_quantmind_storage")
        )

        # Clean up
        if config.storage_dir.exists():
            shutil.rmtree(config.storage_dir)

    def test_existing_directory_handling(self):
        """Test behavior when storage directory already exists."""
        storage_dir = self.temp_dir / "existing_storage"
        storage_dir.mkdir(parents=True, exist_ok=True)

        # Create some existing subdirectories
        (storage_dir / "raw_files").mkdir(exist_ok=True)
        (storage_dir / "custom_subdir").mkdir(exist_ok=True)

        # Initialize config with existing directory
        config = LocalStorageConfig(storage_dir=storage_dir)

        # Should not fail and should create missing subdirectories
        self.assertTrue(config.storage_dir.exists())
        self.assertTrue(config.raw_files_dir.exists())
        self.assertTrue(config.knowledges_dir.exists())
        self.assertTrue(config.embeddings_dir.exists())
        self.assertTrue(config.extra_dir.exists())

        # Custom subdirectory should still exist
        self.assertTrue((storage_dir / "custom_subdir").exists())

    def test_nested_path_creation(self):
        """Test creation of deeply nested storage paths."""
        nested_dir = self.temp_dir / "level1" / "level2" / "level3" / "storage"

        config = LocalStorageConfig(storage_dir=nested_dir)

        # Should create all parent directories
        self.assertTrue(nested_dir.exists())
        self.assertTrue(config.raw_files_dir.exists())
        self.assertTrue(config.knowledges_dir.exists())
        self.assertTrue(config.embeddings_dir.exists())
        self.assertTrue(config.extra_dir.exists())

    def test_string_path_input(self):
        """Test that string paths are properly converted to Path objects."""
        storage_dir_str = str(self.temp_dir / "string_input")

        config = LocalStorageConfig(storage_dir=storage_dir_str)

        # Should convert string to Path and work properly
        self.assertIsInstance(config.storage_dir, Path)
        self.assertTrue(config.storage_dir.exists())
        self.assertTrue(config.raw_files_dir.exists())

    def test_directory_permissions(self):
        """Test that created directories have proper permissions."""
        storage_dir = self.temp_dir / "permission_test"

        config = LocalStorageConfig(storage_dir=storage_dir)

        # Check that directories are readable and writable
        self.assertTrue(storage_dir.is_dir())
        self.assertTrue(config.raw_files_dir.is_dir())
        self.assertTrue(config.knowledges_dir.is_dir())
        self.assertTrue(config.embeddings_dir.is_dir())
        self.assertTrue(config.extra_dir.is_dir())

        # Test that we can write to directories
        test_file = config.raw_files_dir / "test.txt"
        test_file.write_text("test content")
        self.assertTrue(test_file.exists())
        self.assertEqual(test_file.read_text(), "test content")

    def test_model_dump_functionality(self):
        """Test that model can be serialized properly."""
        storage_dir = self.temp_dir / "dump_test"

        config = LocalStorageConfig(storage_dir=storage_dir)

        # Test model_dump
        dumped = config.model_dump()
        self.assertIn("storage_dir", dumped)
        self.assertEqual(Path(dumped["storage_dir"]), storage_dir.resolve())

    def test_multiple_config_instances(self):
        """Test that multiple config instances work independently."""
        dir1 = self.temp_dir / "storage1"
        dir2 = self.temp_dir / "storage2"

        config1 = LocalStorageConfig(storage_dir=dir1)
        config2 = LocalStorageConfig(storage_dir=dir2)

        # Both should exist and be different
        self.assertTrue(config1.storage_dir.exists())
        self.assertTrue(config2.storage_dir.exists())
        self.assertNotEqual(config1.storage_dir, config2.storage_dir)

        # Both should have their own subdirectories
        self.assertTrue(config1.raw_files_dir.exists())
        self.assertTrue(config2.raw_files_dir.exists())
        self.assertNotEqual(config1.raw_files_dir, config2.raw_files_dir)

    def test_reconfiguration(self):
        """Test that config can be updated after creation."""
        storage_dir1 = self.temp_dir / "original"
        storage_dir2 = self.temp_dir / "updated"

        # Create initial config
        config = LocalStorageConfig(storage_dir=storage_dir1)
        self.assertTrue(storage_dir1.exists())

        # Update storage directory
        config.storage_dir = storage_dir2
        config.model_post_init(None)  # Manually trigger post_init

        # New directory should be created
        self.assertTrue(storage_dir2.exists())
        self.assertTrue((storage_dir2 / "raw_files").exists())


if __name__ == "__main__":
    unittest.main()
