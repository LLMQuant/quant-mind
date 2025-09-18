"""Sanity checks for the QuantMind tool decorator and helpers."""

import unittest

from quantmind.tools._function_type_hints_utils import DocstringParsingException
from quantmind.tools.base import Tool, tool, validate_tool_arguments


@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
        x (float): Left operand.
        y (float): Right operand.

    Returns:
        float: Product of the two inputs.
    """
    return x * y


class UppercaseTool(Tool):
    """Simple concrete Tool subclass for serialization checks."""

    name = "uppercase_tool"
    description = "Uppercase string input."
    inputs = {
        "text": {
            "type": "string",
            "description": "Text to uppercase.",
        }
    }
    output_type = "string"

    def forward(self, text: str) -> str:  # noqa: D401 - self explanatory
        return text.upper()


class StructuredTool(Tool):
    """Tool returning structured output with an explicit schema."""

    name = "structured_tool"
    description = "Return the length of a string in a structured payload."
    inputs = {
        "text": {
            "type": "string",
            "description": "Input text to measure.",
        }
    }
    output_type = "object"
    output_schema = {
        "type": "object",
        "properties": {
            "length": {"type": "integer"},
        },
        "required": ["length"],
    }

    def forward(self, text: str) -> dict[str, int]:  # noqa: D401
        return {"length": len(text)}


class ToolDecoratorTests(unittest.TestCase):
    """Validate the lightweight tool decorator behaviour."""

    def test_tool_requires_docstring(self) -> None:
        """Functions without docstrings should be rejected."""

        def missing_doc(a: int) -> int:
            return a

        with self.assertRaises(DocstringParsingException):
            tool(missing_doc)

    def test_tool_metadata_and_execution(self) -> None:
        """Decorated tools expose schema metadata and execute correctly."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two integers.

            Args:
                a (int): First operand.
                b (int): Second operand.

            Returns:
                int: Computed sum.
            """
            return a + b

        self.assertEqual(add.description.strip(), "Add two integers.")
        self.assertEqual(add.inputs["a"]["type"], "integer")

        validate_tool_arguments(add, {"a": 1, "b": 2})
        with self.assertRaises(ValueError):
            validate_tool_arguments(add, {"a": 1})
        with self.assertRaises(TypeError):
            validate_tool_arguments(add, {"a": "one", "b": 2})

        self.assertEqual(add(a=3, b=4), 7)
        self.assertEqual(add({"a": 5, "b": 6}), 11)
        self.assertIsInstance(add, Tool)

    def test_optional_and_enum_inputs(self) -> None:
        """Defaults impact required flag and enum choices validated."""

        @tool
        def choose_action(action: str, mode: str = "auto") -> str:
            """Select an action.

            Args:
                action (str): Action flag (choices: ["buy", "sell"])
                mode (str): Execution mode.
            """
            return f"{mode}:{action}"

        self.assertEqual(
            choose_action.inputs["action"]["enum"], ["buy", "sell"]
        )

        validate_tool_arguments(choose_action, {"action": "buy"})
        with self.assertRaises(ValueError):
            validate_tool_arguments(choose_action, {"mode": "manual"})

    def test_positional_invocation_rules(self) -> None:
        """Only single-argument tools allow positional calls."""

        @tool
        def echo(text: str) -> str:
            """Echo text.

            Args:
                text (str): Text to return.
            """
            return text

        self.assertEqual(echo("hello"), "hello")

        @tool
        def concat(a: str, b: str) -> str:
            """Concatenate strings.

            Args:
                a (str): First part.
                b (str): Second part.
            """
            return a + b

        with self.assertRaises(TypeError):
            concat("value")


class ToolRuntimeTests(unittest.TestCase):
    """Cover behaviour of concrete Tool subclasses and serialization helpers."""

    def test_multiply_tool_schema_and_validation(self) -> None:
        """Decorated module-level tool exposes schema metadata and validation."""
        self.assertEqual(multiply.name, "multiply")
        self.assertEqual(multiply.inputs["x"]["type"], "number")

        validate_tool_arguments(multiply, {"x": 1, "y": 2.5})
        with self.assertRaises(ValueError):
            validate_tool_arguments(multiply, {"x": 1})
        with self.assertRaises(TypeError):
            validate_tool_arguments(multiply, {"x": "bad", "y": 1})

        self.assertEqual(multiply(x=2, y=3), 6)

    def test_structured_tool_prompts_and_call(self) -> None:
        """StructuredTool generates descriptive prompts and handles dict inputs."""
        structured = StructuredTool()
        code_prompt = structured.to_code_prompt()
        self.assertIn("structured_tool", code_prompt)
        self.assertIn(
            "Important: This tool returns structured output!", code_prompt
        )

        calling_prompt = structured.to_tool_calling_prompt()
        self.assertIn("structured_tool", calling_prompt)
        self.assertIn("Returns an output of type: object", calling_prompt)

        result = structured({"text": "alpha"})
        self.assertEqual(result, {"length": 5})
        self.assertTrue(structured.is_initialized)

    def test_subclass_to_dict_roundtrip(self) -> None:
        """Subclass tools serialize to code and can be rehydrated."""
        tool_instance = UppercaseTool()
        tool_dict = tool_instance.to_dict()

        self.assertEqual(tool_dict["name"], "uppercase_tool")
        self.assertIn("uppercase_tool", tool_dict["code"])
        self.assertIn("quantmind", tool_dict["requirements"])

        reloaded = Tool.from_dict(tool_dict)
        self.assertEqual(reloaded(text="abc"), "ABC")

    def test_decorated_tool_to_dict_contains_forward_source(self) -> None:
        """SimpleTool export includes the wrapped forward definition."""
        exported = multiply.to_dict()
        self.assertEqual(exported["name"], "multiply")
        self.assertIn("class SimpleTool", exported["code"])
        self.assertIn("def forward(self, x: float, y: float)", exported["code"])

    def test_invalid_tool_definition_detected_on_init(self) -> None:
        """Invalid class attributes should raise during instantiation."""

        class InvalidNameTool(Tool):
            name = "invalid tool name"
            description = "Bad name"
            inputs = {
                "value": {
                    "type": "string",
                    "description": "v",
                }
            }
            output_type = "string"

            def forward(
                self, value: str
            ) -> str:  # pragma: no cover - never called
                return value

        with self.assertRaises(Exception):
            InvalidNameTool()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
