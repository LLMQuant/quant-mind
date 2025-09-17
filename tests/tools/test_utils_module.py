"""Unit tests covering QuantMind tool utility helpers."""

import ast
import unittest

from quantmind.tools import Tool
from quantmind.tools.utils import (
    ImportFinder,
    get_source,
    instance_to_source,
    is_valid_name,
)


class DummyTool(Tool):
    """Minimal concrete tool used to verify source serialization."""

    name = "dummy_tool"
    description = "Example tool for instance_to_source"
    inputs = {
        "input": {
            "type": "string",
            "description": "Payload text",
            "required": True,
        }
    }
    output_type = "string"
    long_text = "Line one\nLine two"

    def forward(self, input: str) -> str:
        return input.upper()


class UtilsTests(unittest.TestCase):
    """Validate helper behaviours mirroring smolagents utility tests."""

    def test_import_finder_collects_base_modules(self) -> None:
        """Ensure ImportFinder tracks unique top-level package names."""
        code = "import numpy as np\nfrom pandas.core.frame import DataFrame\n"
        finder = ImportFinder()
        finder.visit(ast.parse(code))
        self.assertEqual(finder.packages, {"numpy", "pandas"})

    def test_instance_to_source_includes_base_import(self) -> None:
        """instance_to_source emits base class import and method body."""
        tool_source = instance_to_source(DummyTool(), base_cls=Tool)
        self.assertIn("from quantmind.tools.base import Tool", tool_source)
        self.assertIn("class DummyTool(Tool):", tool_source)
        self.assertIn("def forward(self, input: str) -> str:", tool_source)
        self.assertIn("return input.upper()", tool_source)

    def test_get_source_standard_function(self) -> None:
        """Function source is retrieved and dedented by get_source."""

        def helper(value: int) -> int:
            return value + 1

        expected = "def helper(value: int) -> int:\n    return value + 1"
        self.assertEqual(get_source(helper), expected)

    def test_get_source_rejects_non_callable(self) -> None:
        """get_source raises TypeError for unsupported inputs."""
        with self.assertRaises(TypeError):
            get_source(42)  # type: ignore[arg-type]

    def test_is_valid_name(self) -> None:
        """Names must be valid identifiers and not keywords."""
        self.assertTrue(is_valid_name("valid_name"))
        self.assertFalse(is_valid_name("invalid name"))
        self.assertFalse(is_valid_name("for"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
