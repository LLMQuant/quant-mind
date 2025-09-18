import ast
import unittest
from textwrap import dedent

from quantmind.tools import Tool
from quantmind.tools._tool_validation import (
    MethodChecker,
    validate_tool_attributes,
)

UNDEFINED_VARIABLE = "undefined"


class ValidTool(Tool):
    """Valid tool."""

    name = "valid_tool"
    description = "A valid tool"
    inputs = {
        "input": {
            "type": "string",
            "description": "input payload",
            "required": True,
        }
    }
    output_type = "string"
    simple_attr = "string"
    dict_attr = {"key": "value"}

    def __init__(self, optional_param: str = "default") -> None:
        super().__init__()
        self.param = optional_param

    def forward(self, input: str) -> str:
        return input.upper()


class InvalidToolName(Tool):
    """Invalid tool name."""

    name = "invalid tool name"
    description = "Tool with invalid name"
    inputs = {
        "input": {
            "type": "string",
            "description": "input payload",
            "required": True,
        }
    }
    output_type = "string"

    def forward(self, input: str) -> str:
        return input


class InvalidToolComplexAttrs(Tool):
    """Invalid tool complex attributes."""

    name = "invalid_tool"
    description = "Tool with complex class attributes"
    inputs = {
        "input": {
            "type": "string",
            "description": "input payload",
            "required": True,
        }
    }
    output_type = "string"
    complex_attr = [x for x in range(3)]

    def forward(self, input: str) -> str:
        return input


class InvalidToolRequiredParams(Tool):
    """Invalid tool required parameters."""

    name = "invalid_tool"
    description = "Tool with required params"
    inputs = {
        "input": {
            "type": "string",
            "description": "input payload",
            "required": True,
        }
    }
    output_type = "string"

    def __init__(self, required_param: str, kwarg1: int = 1) -> None:
        super().__init__()
        self.param = required_param

    def forward(self, input: str) -> str:
        return input


class InvalidToolNonLiteralDefaultParam(Tool):
    """Invalid tool non-literal default parameter."""

    name = "invalid_tool"
    description = "Tool with non-literal default parameter value"
    inputs = {
        "input": {
            "type": "string",
            "description": "input payload",
            "required": True,
        }
    }
    output_type = "string"

    def __init__(self, default_param: str = UNDEFINED_VARIABLE) -> None:
        super().__init__()
        self.default_param = default_param

    def forward(self, input: str) -> str:
        return input


class InvalidToolUndefinedNames(Tool):
    """Invalid tool undefined names."""

    name = "invalid_tool"
    description = "Tool with undefined names"
    inputs = {
        "input": {
            "type": "string",
            "description": "input payload",
            "required": True,
        }
    }
    output_type = "string"

    def forward(self, input: str) -> str:
        return UNDEFINED_VARIABLE


class MultipleAssignmentsTool(Tool):
    """Multiple assignments tool."""

    name = "multiple_assignments_tool"
    description = "Tool with multiple assignments"
    inputs = {
        "input": {
            "type": "string",
            "description": "input payload",
            "required": True,
        }
    }
    output_type = "string"

    def forward(self, input: str) -> str:
        first, second = "1", "2"
        return input + first + second


class TestToolValidation(unittest.TestCase):
    """Test tool validation."""

    def test_validate_tool_attributes_valid(self) -> None:
        self.assertIsNone(validate_tool_attributes(ValidTool))

    def test_invalid_tool_name(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Class attribute 'name' must be a valid Python identifier",
        ):
            validate_tool_attributes(InvalidToolName)

    def test_complex_class_attribute(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Complex attributes should be defined in __init__"
        ):
            validate_tool_attributes(InvalidToolComplexAttrs)

    def test_required_init_parameter(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Parameters in __init__ must have default values"
        ):
            validate_tool_attributes(InvalidToolRequiredParams)

    def test_non_literal_default(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Parameters in __init__ must have literal default values",
        ):
            validate_tool_attributes(InvalidToolNonLiteralDefaultParam)

    def test_undefined_names(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Name 'UNDEFINED_VARIABLE' is undefined"
        ):
            validate_tool_attributes(InvalidToolUndefinedNames)

    def test_multiple_assignments_allowed(self) -> None:
        self.assertIsNone(validate_tool_attributes(MultipleAssignmentsTool))


class TestMethodChecker(unittest.TestCase):
    """Test method checker."""

    def test_multiple_assignments(self) -> None:
        source_code = dedent(
            """
            def forward(self) -> str:
                a, b = "1", "2"
                return a + b
            """
        )
        method_checker = MethodChecker(set())
        method_checker.visit(ast.parse(source_code))
        self.assertEqual(method_checker.errors, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
