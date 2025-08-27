"""Tests for LLMBlock."""

import json
import unittest
from unittest.mock import Mock, patch

from quantmind.config.llm import LLMConfig
from quantmind.llm.block import LLMBlock, create_llm_block


class TestLLMBlock(unittest.TestCase):
    """Test cases for LLMBlock."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000,
            api_key="test-key",
            timeout=1,
            retry_delay=0.01,
        )

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_init_success(self, mock_litellm):
        """Test successful initialization."""
        block = LLMBlock(self.config)

        self.assertEqual(block.config, self.config)
        mock_litellm.set_verbose = False
        self.assertEqual(mock_litellm.num_retries, 3)
        self.assertEqual(mock_litellm.request_timeout, 1)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", False)
    def test_init_litellm_unavailable(self):
        """Test initialization when LiteLLM is not available."""
        with self.assertRaises(ImportError) as context:
            LLMBlock(self.config)

        self.assertIn("LiteLLM is not available", str(context.exception))

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("os.environ", {})
    def test_setup_litellm_openai(self, mock_litellm):
        """Test LiteLLM setup for OpenAI."""
        config = LLMConfig(model="gpt-4o", api_key="test-key")

        with patch("os.environ", {}) as mock_env:
            block = LLMBlock(config)
            self.assertEqual(mock_env.get("OPENAI_API_KEY"), "test-key")

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_generate_text_success(self, mock_completion, mock_litellm):
        """Test successful text generation."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_completion.return_value = mock_response

        block = LLMBlock(self.config)
        result = block.generate_text("Test prompt")

        self.assertEqual(result, "Hello, world!")
        mock_completion.assert_called_once()

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_generate_text_with_system_prompt(
        self, mock_completion, mock_litellm
    ):
        """Test text generation with system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response"
        mock_completion.return_value = mock_response

        block = LLMBlock(self.config)
        result = block.generate_text(
            "Test prompt", system_prompt="You are a helpful assistant."
        )

        # Check that system prompt was included in messages
        call_args = mock_completion.call_args
        messages = call_args[1]["messages"]

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a helpful assistant.")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Test prompt")

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_generate_text_failure(self, mock_completion, mock_litellm):
        """Test text generation failure."""
        mock_completion.side_effect = Exception("API Error")

        block = LLMBlock(self.config)
        result = block.generate_text("Test prompt")

        self.assertIsNone(result)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_generate_structured_output_success(
        self, mock_completion, mock_litellm
    ):
        """Test successful structured output generation."""
        json_response = {"key": "value", "number": 42}
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(json_response)
        mock_completion.return_value = mock_response

        block = LLMBlock(self.config)
        result = block.generate_structured_output("Generate JSON")

        self.assertEqual(result, json_response)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_generate_structured_output_invalid_json(
        self, mock_completion, mock_litellm
    ):
        """Test structured output with invalid JSON."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_completion.return_value = mock_response

        block = LLMBlock(self.config)
        result = block.generate_structured_output("Generate JSON")

        self.assertIsNone(result)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_build_messages_basic(self, mock_litellm):
        """Test basic message building."""
        block = LLMBlock(self.config)
        messages = block._build_messages("Hello")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_build_messages_with_system_prompt(self, mock_litellm):
        """Test message building with system prompt."""
        block = LLMBlock(self.config)
        messages = block._build_messages(
            "Hello", system_prompt="You are helpful"
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are helpful")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Hello")

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_build_messages_with_custom_instructions(self, mock_litellm):
        """Test message building with custom instructions."""
        config = LLMConfig(model="gpt-4o", custom_instructions="Be concise")
        block = LLMBlock(config)
        messages = block._build_messages("Hello")

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("Hello", messages[0]["content"])
        self.assertIn("Be concise", messages[0]["content"])

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_call_with_retry_success(self, mock_completion, mock_litellm):
        """Test successful call with retry."""
        mock_response = Mock()
        mock_completion.return_value = mock_response

        block = LLMBlock(self.config)
        result = block._call_with_retry({"model": "gpt-4o"})

        self.assertEqual(result, mock_response)
        mock_completion.assert_called_once()

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    @patch("time.sleep")
    def test_call_with_retry_failure_then_success(
        self, mock_sleep, mock_completion, mock_litellm
    ):
        """Test retry logic with failure then success."""
        mock_response = Mock()
        mock_completion.side_effect = [
            Exception("First failure"),
            mock_response,
        ]

        block = LLMBlock(self.config)
        result = block._call_with_retry({"model": "gpt-4o"})

        self.assertEqual(result, mock_response)
        self.assertEqual(mock_completion.call_count, 2)
        mock_sleep.assert_called_once_with(0.01)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    @patch("time.sleep")
    def test_call_with_retry_all_failures(
        self, mock_sleep, mock_completion, mock_litellm
    ):
        """Test retry logic with all failures."""
        mock_completion.side_effect = Exception("Always fails")

        block = LLMBlock(self.config)
        result = block._call_with_retry({"model": "gpt-4o"})

        self.assertIsNone(result)
        self.assertEqual(mock_completion.call_count, 4)  # 1 initial + 3 retries
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_extract_json_from_text_success(self, mock_litellm):
        """Test JSON extraction from text."""
        block = LLMBlock(self.config)

        text = 'Here is some JSON: {"key": "value", "number": 42}'
        result = block._extract_json_from_text(text)

        self.assertEqual(result, {"key": "value", "number": 42})

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_extract_json_from_text_failure(self, mock_litellm):
        """Test JSON extraction failure."""
        block = LLMBlock(self.config)

        text = "This is just plain text with no JSON"
        result = block._extract_json_from_text(text)

        self.assertIsNone(result)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_get_info(self, mock_litellm):
        """Test getting block info."""
        block = LLMBlock(self.config)
        info = block.get_info()

        expected_info = {
            "model": "gpt-4o",
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            "timeout": 1,
            "retry_attempts": 3,
        }

        self.assertEqual(info, expected_info)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_update_config(self, mock_litellm):
        """Test configuration update."""
        block = LLMBlock(self.config)

        # Check initial config
        self.assertEqual(block.config.temperature, 0.7)
        self.assertEqual(block.config.max_tokens, 1000)

        # Update config
        block.update_config(temperature=0.5, max_tokens=2000)

        # Check updated config
        self.assertEqual(block.config.temperature, 0.5)
        self.assertEqual(block.config.max_tokens, 2000)
        # Other values should remain unchanged
        self.assertEqual(block.config.model, "gpt-4o")

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_temporary_config(self, mock_litellm):
        """Test temporary configuration context manager."""
        block = LLMBlock(self.config)

        # Check initial config
        self.assertEqual(block.config.temperature, 0.7)

        # Use temporary config
        with block.temporary_config(temperature=0.5):
            self.assertEqual(block.config.temperature, 0.5)

        # Check config is restored
        self.assertEqual(block.config.temperature, 0.7)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_test_connection_success(self, mock_completion, mock_litellm):
        """Test successful connection test."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "OK"
        mock_completion.return_value = mock_response

        block = LLMBlock(self.config)
        result = block.test_connection()

        self.assertTrue(result)

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    @patch("quantmind.llm.block.completion")
    def test_test_connection_failure(self, mock_completion, mock_litellm):
        """Test connection test failure."""
        mock_completion.side_effect = Exception("Connection failed")

        block = LLMBlock(self.config)
        result = block.test_connection()

        self.assertFalse(result)


class TestCreateLLMBlock(unittest.TestCase):
    """Test cases for create_llm_block function."""

    @patch("quantmind.llm.block.LITELLM_AVAILABLE", True)
    @patch("quantmind.llm.block.litellm")
    def test_create_llm_block(self, mock_litellm):
        """Test LLMBlock creation."""
        config = LLMConfig(model="gpt-4o")
        block = create_llm_block(config)

        self.assertIsInstance(block, LLMBlock)
        self.assertEqual(block.config, config)


if __name__ == "__main__":
    unittest.main()
