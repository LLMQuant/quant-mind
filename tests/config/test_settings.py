"""Unit tests for settings configuration system."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from quantmind.config.llm import LLMConfig
from quantmind.config.parsers import LlamaParserConfig, PDFParserConfig
from quantmind.config.settings import (
    Setting,
    create_default_config,
    load_config,
)
from quantmind.config.sources import ArxivSourceConfig
from quantmind.config.storage import LocalStorageConfig
from quantmind.config.taggers import LLMTaggerConfig


class TestSetting(unittest.TestCase):
    """Comprehensive test cases for Setting configuration system."""

    def test_default_setting(self):
        """Test creating Setting with default values."""
        setting = Setting()

        self.assertEqual(setting.log_level, "INFO")
        self.assertIsNone(setting.source)
        self.assertIsNone(setting.parser)
        self.assertIsNone(setting.tagger)
        self.assertIsInstance(setting.storage, LocalStorageConfig)
        self.assertIsInstance(setting.llm, LLMConfig)

    def test_setting_with_components(self):
        """Test creating Setting with component configurations."""
        # Create component configs
        source_config = ArxivSourceConfig(max_results=50)
        parser_config = PDFParserConfig(method="pdfplumber")
        tagger_config = LLMTaggerConfig(max_tags=10)

        setting = Setting(
            source=source_config, parser=parser_config, tagger=tagger_config
        )

        self.assertIsInstance(setting.source, ArxivSourceConfig)
        self.assertEqual(setting.source.max_results, 50)

        self.assertIsInstance(setting.parser, PDFParserConfig)
        self.assertEqual(setting.parser.method, "pdfplumber")

        self.assertIsInstance(setting.tagger, LLMTaggerConfig)
        self.assertEqual(setting.tagger.max_tags, 10)

    def test_parse_config_with_components(self):
        """Test parsing configuration dictionary with various components."""
        config_dict = {
            "source": {
                "type": "arxiv",
                "config": {"max_results": 50, "sort_by": "relevance"},
            },
            "parser": {
                "type": "pdf",
                "config": {
                    "method": "pdfplumber",
                    "download_pdfs": True,
                    "max_file_size_mb": 25,
                },
            },
            "tagger": {
                "type": "llm",
                "config": {"max_tags": 8, "model": "gpt-4o"},
            },
            "log_level": "DEBUG",
        }

        setting = Setting._parse_config(config_dict)

        # Test source parsing
        self.assertIsInstance(setting.source, ArxivSourceConfig)
        self.assertEqual(setting.source.max_results, 50)
        self.assertEqual(setting.source.sort_by, "relevance")

        # Test parser parsing
        self.assertIsInstance(setting.parser, PDFParserConfig)
        self.assertEqual(setting.parser.method, "pdfplumber")
        self.assertTrue(setting.parser.download_pdfs)
        self.assertEqual(setting.parser.max_file_size_mb, 25)

        # Test tagger parsing
        self.assertIsInstance(setting.tagger, LLMTaggerConfig)
        self.assertEqual(setting.tagger.max_tags, 8)
        self.assertEqual(setting.tagger.llm_config.model, "gpt-4o")

        # Test simple fields
        self.assertEqual(setting.log_level, "DEBUG")

        if setting.storage.storage_dir.exists():
            shutil.rmtree(setting.storage.storage_dir)

    def test_parse_config_unknown_types(self):
        """Test parsing configuration with unknown component types."""
        config_dict = {
            "source": {
                "type": "unknown_source",
                "config": {"some_param": "value"},
            },
            "parser": {"type": "llama", "config": {"result_type": "markdown"}},
        }

        setting = Setting._parse_config(config_dict)

        # Unknown source should be ignored
        self.assertIsNone(setting.source)

        # Known parser should be parsed
        self.assertIsInstance(setting.parser, LlamaParserConfig)
        self.assertEqual(setting.parser.result_type, "markdown")

    def test_create_default_config(self):
        """Test creating default configuration."""
        setting = create_default_config()

        # Test default source
        self.assertIsInstance(setting.source, ArxivSourceConfig)
        self.assertEqual(setting.source.max_results, 100)
        self.assertEqual(setting.source.sort_by, "submittedDate")
        self.assertEqual(setting.source.sort_order, "descending")

        # Test default parser
        self.assertIsInstance(setting.parser, PDFParserConfig)
        self.assertEqual(setting.parser.method, "pymupdf")
        self.assertTrue(setting.parser.download_pdfs)
        self.assertTrue(setting.parser.extract_tables)

        # Test default storage
        self.assertIsInstance(setting.storage, LocalStorageConfig)

        # Test default values
        self.assertEqual(setting.log_level, "INFO")
        self.assertIsInstance(setting.llm, LLMConfig)

    def test_load_config_yaml_file(self):
        """Test loading configuration from YAML file."""
        config_dict = {
            "source": {
                "type": "arxiv",
                "config": {"max_results": 25, "sort_by": "relevance"},
            },
            "parser": {
                "type": "pdf",
                "config": {"method": "pdfplumber", "download_pdfs": False},
            },
            "log_level": "WARNING",
        }

        # Test the actual YAML loading with temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            import yaml

            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            setting = load_config(temp_path)

            # Verify loaded configuration
            self.assertIsInstance(setting.source, ArxivSourceConfig)
            self.assertEqual(setting.source.max_results, 25)
            self.assertEqual(setting.source.sort_by, "relevance")

            self.assertIsInstance(setting.parser, PDFParserConfig)
            self.assertEqual(setting.parser.method, "pdfplumber")
            self.assertFalse(setting.parser.download_pdfs)

            self.assertEqual(setting.log_level, "WARNING")

        finally:
            # Clean up
            os.unlink(temp_path)

    def test_load_config_nonexistent_file(self):
        """Test loading configuration from nonexistent file."""
        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_substitute_env_vars(self):
        """Test environment variable substitution in configuration."""
        # Set up environment variables
        os.environ["TEST_VAR"] = "test_value"
        os.environ["API_KEY"] = "secret_key"

        config_dict = {
            "source": {
                "type": "arxiv",
                "config": {
                    "api_key": "${API_KEY}",
                    "max_results": "${MAX_RESULTS:50}",  # with default
                },
            },
        }

        result = Setting.substitute_env_vars(config_dict)

        # Test substitution
        self.assertEqual(result["source"]["config"]["api_key"], "secret_key")
        self.assertEqual(
            result["source"]["config"]["max_results"], "50"
        )  # default used

        # Clean up
        del os.environ["TEST_VAR"]
        del os.environ["API_KEY"]

    def test_substitute_env_vars_nested(self):
        """Test environment variable substitution in nested structures."""
        os.environ["NESTED_VAR"] = "nested_value"

        config_dict = {
            "components": {
                "parser": {
                    "config": {
                        "nested_list": ["${NESTED_VAR}", "static_value"],
                        "nested_dict": {"key": "${NESTED_VAR}"},
                    }
                }
            }
        }

        result = Setting.substitute_env_vars(config_dict)

        self.assertEqual(
            result["components"]["parser"]["config"]["nested_list"][0],
            "nested_value",
        )
        self.assertEqual(
            result["components"]["parser"]["config"]["nested_dict"]["key"],
            "nested_value",
        )

        # Clean up
        del os.environ["NESTED_VAR"]

    @patch.dict(os.environ, {}, clear=True)
    def test_substitute_env_vars_defaults(self):
        """Test environment variable substitution with defaults when vars don't exist."""
        config_dict = {
            "api_key": "${MISSING_KEY:default_key}",
            "no_default": "${MISSING_NO_DEFAULT}",
            "mixed": "prefix_${MISSING_WITH_DEFAULT:default}_suffix",
        }

        result = Setting.substitute_env_vars(config_dict)

        self.assertEqual(result["api_key"], "default_key")
        self.assertEqual(
            result["no_default"], ""
        )  # empty string when no default
        self.assertEqual(result["mixed"], "prefix_default_suffix")

    def test_export_config(self):
        """Test exporting configuration to dictionary."""
        setting = Setting(
            source=ArxivSourceConfig(max_results=30),
            parser=PDFParserConfig(method="pdfplumber", download_pdfs=True),
            tagger=LLMTaggerConfig(max_tags=5),
            log_level="DEBUG",
        )

        config_dict = setting._export_config()

        # Test component export
        self.assertEqual(config_dict["source"]["type"], "arxiv")
        self.assertEqual(config_dict["source"]["config"]["max_results"], 30)

        self.assertEqual(config_dict["parser"]["type"], "pdf")
        self.assertEqual(
            config_dict["parser"]["config"]["method"], "pdfplumber"
        )
        self.assertTrue(config_dict["parser"]["config"]["download_pdfs"])

        self.assertEqual(config_dict["tagger"]["type"], "llm")
        self.assertEqual(config_dict["tagger"]["config"]["max_tags"], 5)

        # Test simple fields
        self.assertEqual(config_dict["log_level"], "DEBUG")

        # Test sensitive data exclusion
        self.assertNotIn("api_key", config_dict["llm"])

        assert setting.storage.storage_dir.exists()
        if setting.storage.storage_dir.exists():
            shutil.rmtree(setting.storage.storage_dir)
        assert not setting.storage.storage_dir.exists()

    def test_save_to_yaml(self):
        """Test saving configuration to YAML file."""
        setting = Setting(
            source=ArxivSourceConfig(max_results=20),
            parser=PDFParserConfig(method="pymupdf"),
        )

        # Test saving to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            temp_path = f.name

        try:
            setting.save_to_yaml(temp_path)

            # Verify file was created and contains expected content
            with open(temp_path, "r") as f:
                import yaml

                saved_config = yaml.safe_load(f)

            self.assertEqual(saved_config["source"]["type"], "arxiv")
            self.assertEqual(
                saved_config["source"]["config"]["max_results"], 20
            )
            self.assertEqual(saved_config["parser"]["type"], "pdf")
            self.assertEqual(
                saved_config["parser"]["config"]["method"], "pymupdf"
            )

        finally:
            # Clean up
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
