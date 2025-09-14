import asyncio
import unittest

from pydantic import ValidationError

from quantmind.tools import BaseTool, tool


class TestTools(unittest.TestCase):
    """Test the tools module."""

    def test_tool_requires_docstring(self):
        """Test that a tool requires a docstring."""

        def no_doc(a: int):
            return a

        with self.assertRaises(ValueError):
            tool(no_doc)

    def test_sync_function_tool_run_and_schema(self):
        """Test that a sync function tool runs and validates the schema."""

        @tool
        def add(a: int, b: int) -> int:
            """Adds two integers."""
            return a + b

        # to_openai_schema shape
        schema = add.to_openai_schema()
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "add")
        self.assertIn("parameters", schema["function"])

        # args_schema validation
        with self.assertRaises(ValidationError):
            # missing required field
            asyncio.run(add.run(a=1))

        result = asyncio.run(add.run(a=2, b=3))
        self.assertEqual(result, 5)

    def test_async_function_tool_run(self):
        """Test that an async function tool runs."""

        @tool
        async def mul(a: int, b: int) -> int:
            """Multiplies two integers asynchronously."""
            return a * b

        result = asyncio.run(mul.run(a=4, b=5))
        self.assertEqual(result, 20)


if __name__ == "__main__":
    unittest.main()
