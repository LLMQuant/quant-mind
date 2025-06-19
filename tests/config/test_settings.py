"""Unit tests for settings configuration system."""

import unittest
from unittest.mock import patch, mock_open

from quantmind.config.parsers import LlamaParserConfig, PDFParserConfig
from quantmind.config.settings import (
    ComponentConfig,
    Settings,
    create_default_config,
    load_config,
)


class TestComponentConfig(unittest.TestCase):
    """Test cases for ComponentConfig."""

    def test_component_config_creation(self):
        """Test creating ComponentConfig with different config types."""
        # Test with Pydantic config
        pydantic_config = LlamaParserConfig(
            result_type="markdown", parsing_mode="fast"
        )
        component = ComponentConfig(
            name="llama", type="LlamaParser", config=pydantic_config
        )

        self.assertEqual(component.name, "llama")
        self.assertEqual(component.type, "LlamaParser")
        self.assertIsInstance(component.config, LlamaParserConfig)
        self.assertTrue(component.enabled)

        # Test with dict config
        dict_config = {"method": "pymupdf", "download_pdfs": True}
        component = ComponentConfig(
            name="pdf", type="PDFParser", config=dict_config
        )

        self.assertEqual(component.name, "pdf")
        self.assertEqual(component.type, "PDFParser")
        self.assertEqual(component.config, dict_config)


class TestSettings(unittest.TestCase):
    """Test cases for Settings."""

    def test_default_settings(self):
        """Test default settings creation."""
        settings = Settings()

        self.assertEqual(settings.log_level, "INFO")
        self.assertEqual(settings.data_dir, "./data")
        self.assertEqual(settings.temp_dir, "/tmp")
        self.assertEqual(len(settings.sources), 0)
        self.assertEqual(len(settings.parsers), 0)

    def test_from_dict_with_parsers(self):
        """Test creating Settings from dict with parser configurations."""
        config_dict = {
            "parsers": {
                "llama": {
                    "type": "LlamaParser",
                    "config": {
                        "result_type": "markdown",
                        "parsing_mode": "premium",
                        "max_file_size_mb": 25,
                    },
                    "enabled": True,
                },
                "pdf": {
                    "type": "PDFParser",
                    "config": {
                        "method": "pdfplumber",
                        "download_pdfs": False,
                        "max_file_size_mb": 30,
                    },
                    "enabled": True,
                },
                "unknown": {
                    "type": "UnknownParser",
                    "config": {"custom_setting": "value"},
                    "enabled": False,
                },
            }
        }

        settings = Settings.from_dict(config_dict)

        # Test LlamaParser config
        llama_parser = settings.parsers["llama"]
        self.assertEqual(llama_parser.type, "LlamaParser")
        self.assertIsInstance(llama_parser.config, LlamaParserConfig)
        self.assertEqual(llama_parser.config.result_type, "markdown")
        self.assertEqual(llama_parser.config.parsing_mode, "premium")
        self.assertEqual(llama_parser.config.max_file_size_mb, 25)

        # Test PDFParser config
        pdf_parser = settings.parsers["pdf"]
        self.assertEqual(pdf_parser.type, "PDFParser")
        self.assertIsInstance(pdf_parser.config, PDFParserConfig)
        self.assertEqual(pdf_parser.config.method, "pdfplumber")
        self.assertFalse(pdf_parser.config.download_pdfs)
        self.assertEqual(pdf_parser.config.max_file_size_mb, 30)

        # Test unknown parser (should keep as dict)
        unknown_parser = settings.parsers["unknown"]
        self.assertEqual(unknown_parser.type, "UnknownParser")
        self.assertIsInstance(unknown_parser.config, dict)
        self.assertEqual(unknown_parser.config["custom_setting"], "value")
        self.assertFalse(unknown_parser.enabled)

    def test_to_dict_with_parsers(self):
        """Test converting Settings to dict with parser configurations."""
        settings = Settings()

        # Add parser configurations
        settings.parsers["llama"] = ComponentConfig(
            name="llama",
            type="LlamaParser",
            config=LlamaParserConfig(
                result_type="text", parsing_mode="fast", max_file_size_mb=40
            ),
        )

        settings.parsers["pdf"] = ComponentConfig(
            name="pdf",
            type="PDFParser",
            config=PDFParserConfig(method="pymupdf", download_pdfs=True),
        )

        config_dict = settings.to_dict()

        # Test LlamaParser in dict
        llama_config = config_dict["parsers"]["llama"]
        self.assertEqual(llama_config["type"], "LlamaParser")
        self.assertEqual(llama_config["config"]["result_type"], "text")
        self.assertEqual(llama_config["config"]["parsing_mode"], "fast")
        self.assertEqual(llama_config["config"]["max_file_size_mb"], 40)

        # Test PDFParser in dict
        pdf_config = config_dict["parsers"]["pdf"]
        self.assertEqual(pdf_config["type"], "PDFParser")
        self.assertEqual(pdf_config["config"]["method"], "pymupdf")
        self.assertTrue(pdf_config["config"]["download_pdfs"])

    def test_get_enabled_parsers(self):
        """Test getting enabled parsers."""
        settings = Settings()

        # Add enabled and disabled parsers
        settings.parsers["enabled"] = ComponentConfig(
            name="enabled",
            type="LlamaParser",
            config=LlamaParserConfig(),
            enabled=True,
        )

        settings.parsers["disabled"] = ComponentConfig(
            name="disabled",
            type="PDFParser",
            config=PDFParserConfig(),
            enabled=False,
        )

        enabled_parsers = settings.get_enabled_parsers()

        self.assertEqual(len(enabled_parsers), 1)
        self.assertIn("enabled", enabled_parsers)
        self.assertNotIn("disabled", enabled_parsers)


class TestDefaultConfig(unittest.TestCase):
    """Test cases for default configuration creation."""

    def test_create_default_config(self):
        """Test creating default configuration."""
        settings = create_default_config()

        # Should have default parsers
        self.assertIn("pdf", settings.parsers)
        self.assertIn("llama", settings.parsers)

        # Test PDF parser config
        pdf_parser = settings.parsers["pdf"]
        self.assertEqual(pdf_parser.type, "PDFParser")
        self.assertIsInstance(pdf_parser.config, PDFParserConfig)
        self.assertEqual(pdf_parser.config.method, "pymupdf")
        self.assertTrue(pdf_parser.enabled)

        # Test Llama parser config
        llama_parser = settings.parsers["llama"]
        self.assertEqual(llama_parser.type, "LlamaParser")
        self.assertIsInstance(llama_parser.config, LlamaParserConfig)
        self.assertEqual(llama_parser.config.result_type, "markdown")
        self.assertFalse(llama_parser.enabled)  # Disabled by default


class TestConfigLoading(unittest.TestCase):
    """Test cases for configuration loading."""

    @patch("quantmind.utils.env.EnvConfig.get_env_var")
    @patch("quantmind.utils.env.EnvConfig.load_dotenv")
    @patch("builtins.open", new_callable=mock_open)
    @patch("quantmind.config.settings.yaml.safe_load")
    @patch("quantmind.config.settings.Path.exists")
    def test_load_config_yaml(
        self,
        mock_exists,
        mock_yaml_load,
        mock_file,
        mock_load_dotenv,
        mock_get_env_var,
    ):
        """Test loading configuration from YAML file."""
        mock_exists.return_value = True
        mock_load_dotenv.return_value = True
        mock_get_env_var.return_value = None  # No environment overrides
        mock_yaml_load.return_value = {
            "parsers": {
                "llama": {
                    "type": "LlamaParser",
                    "config": {"result_type": "markdown"},
                }
            }
        }

        settings = load_config("test_config.yaml")

        mock_file.assert_called_once()
        mock_yaml_load.assert_called_once()
        self.assertIn("llama", settings.parsers)


if __name__ == "__main__":
    unittest.main()
