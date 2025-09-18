"""Tool system implementation for QuantMind.

This module provides the core tool infrastructure, primarily based on the Smolagents
implementation (@tools.py) as a starting point. It includes base classes, decorators,
and utilities for creating and managing tools within the QuantMind framework.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import sys
import textwrap
import types
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from ._function_type_hints_utils import (
    TypeHintParsingException,
    _get_json_schema_type,
    get_imports,
    get_json_schema,
)
from ._tool_validation import MethodChecker, validate_tool_attributes
from .utils import (
    BASE_BUILTIN_MODULES,
    get_source,
    instance_to_source,
    is_valid_name,
)

logger = logging.getLogger(__name__)


def validate_after_init(cls):
    """A class decorator that automatically validates tool arguments after initialization.

    This decorator wraps the class's __init__ method to call validate_arguments()
    immediately after the original initialization is complete. This ensures that
    any tool instance is validated upon creation without requiring manual validation calls.

    Args:
        cls: The class to be decorated (typically a Tool subclass)

    Returns:
        The decorated class with automatic post-init validation
    """
    original_init = cls.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.validate_arguments()

    cls.__init__ = new_init
    return cls


AUTHORIZED_TYPES = [
    "string",
    "boolean",
    "integer",
    "number",
    "image",
    "audio",
    "array",
    "object",
    "any",
    "null",
]

CONVERSION_DICT = {"str": "string", "int": "integer", "float": "number"}


class BaseTool(ABC):
    name: str

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Any:
        pass


class Tool(BaseTool):
    """A base class for the functions used by the agent.

    Subclass this and implement the `forward` method as well as the following
    class attributes.

    Attributes:
        description (str): A short description of what your tool does, the
            inputs it expects and the output(s) it will return. For instance
            'This is a tool that downloads a file from a `url`. It takes the
            `url` as input, and returns the text contained in the file'.
        name (str): A performative name that will be used for your tool in
            the prompt to the agent. For instance `"text-classifier"` or
            `"image_generator"`.
        inputs (Dict[str, Dict[str, Union[str, type, bool]]]): The dict of
            modalities expected for the inputs. It has one `type` key and a
            `description` key. This is used by `launch_gradio_demo` or to make
            a nice space from your tool, and also can be used in the generated
            description for your tool.
        output_type (type): The type of the tool output. This is used by
            `launch_gradio_demo` or to make a nice space from your tool, and
            also can be used in the generated description for your tool.
        output_schema (Dict[str, Any], optional): The JSON schema defining the
            expected structure of the tool output. This can be included in
            system prompts to help agents understand the expected output
            format. Note: This is currently used for informational purposes
            only and does not perform actual output validation.

    Note:
        You can also override the method [`~Tool.setup`] if your tool has an
        expensive operation to perform before being usable (such as loading a
        model). [`~Tool.setup`] will be called the first time you use your
        tool, but not at instantiation.
    """

    name: str
    description: str
    inputs: dict[str, dict[str, str | type | bool]]
    output_type: str
    output_schema: dict[str, Any] | None = None

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        validate_after_init(cls)

    def validate_arguments(self):
        """Validate the tool's arguments.

        This method validates the tool's arguments to ensure they are of the
        correct type and format.
        """
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": dict,
            "output_type": str,
        }
        # Validate class attributes
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if attr_value is None:
                raise TypeError(f"You must set an attribute {attr}.")
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Attribute {attr} should have type {expected_type.__name__}, got {type(attr_value)} instead."
                )

        # Validate optional output_schema attribute
        output_schema = getattr(self, "output_schema", None)
        if output_schema is not None and not isinstance(output_schema, dict):
            raise TypeError(
                f"Attribute output_schema should have type dict, got {type(output_schema)} instead."
            )

        # - Validate name
        if not is_valid_name(self.name):
            raise Exception(
                f"Invalid Tool name '{self.name}': must be a valid Python identifier and not a reserved keyword"
            )

        # Validate inputs
        for input_name, input_content in self.inputs.items():
            assert isinstance(
                input_content, dict
            ), f"Input '{input_name}' should be a dictionary."
            assert (
                "type" in input_content and "description" in input_content
            ), f"Input '{input_name}' should have keys 'type' and 'description', has only {list(input_content.keys())}."
            # Get input_types as a list, whether from a string or list
            if isinstance(input_content["type"], str):
                input_types = [input_content["type"]]
            elif isinstance(input_content["type"], list):
                input_types = input_content["type"]
                # Check if all elements are strings
                if not all(isinstance(t, str) for t in input_types):
                    raise TypeError(
                        f"Input '{input_name}': when type is a list, all elements must be strings, got {input_content['type']}"
                    )
            else:
                raise TypeError(
                    f"Input '{input_name}': type must be a string or list of strings, got {type(input_content['type']).__name__}"
                )
            # Check all types are authorized
            invalid_types = [
                t for t in input_types if t not in AUTHORIZED_TYPES
            ]
            if invalid_types:
                raise ValueError(
                    f"Input '{input_name}': types {invalid_types} must be one of {AUTHORIZED_TYPES}"
                )
        # Validate output type
        assert getattr(self, "output_type", None) in AUTHORIZED_TYPES

    def forward(self, *args, **kwargs):
        """Implement the forward method in your subclass of `Tool`."""
        raise NotImplementedError(
            "Write this method in your subclass of `Tool`."
        )

    def __call__(self, *args, **kwargs):
        """Call the tool.

        This method calls the tool's forward method with the given arguments.
        """
        if not self.is_initialized:
            self.setup()

        # Handle the arguments might be passed as a single dictionary
        if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
            potential_kwargs = args[0]

            # If the dictionary keys match our input parameters, convert it to kwargs
            if all(key in self.inputs for key in potential_kwargs):
                args = ()
                kwargs = potential_kwargs

        outputs = self.forward(*args, **kwargs)
        return outputs

    def setup(self):
        """Setup the tool.

        Overwrite this method here for any operation that is expensive and needs to be executed
        before you start using your tool. Such as loading a big model.
        """
        self.is_initialized = True

    def to_code_prompt(self) -> str:
        """TODO: Add docstring & example for `to_code_prompt` function."""
        args_signature = ", ".join(
            f"{arg_name}: {arg_schema['type']}"
            for arg_name, arg_schema in self.inputs.items()
        )

        # Use dict type for tools with output schema to indicate structured return
        has_schema = (
            hasattr(self, "output_schema") and self.output_schema is not None
        )
        output_type = "dict" if has_schema else self.output_type
        tool_signature = f"({args_signature}) -> {output_type}"
        tool_doc = self.description

        # Add an important note for smaller models (e.g. Mistral Small, Gemma 3, etc.)
        # to properly handle structured output.
        if has_schema:
            IMPORTANT_NOTE_FOR_SMALL_MODELS = "Important: This tool returns structured output! Use the JSON schema below to directly access fields like result['field_name']. NO print() statements needed to inspect the output!"
            tool_doc += "\n\n" + IMPORTANT_NOTE_FOR_SMALL_MODELS

        # Add arguments documentation
        if self.inputs:
            args_descriptions = "\n".join(
                f"{arg_name}: {arg_schema['description']}"
                for arg_name, arg_schema in self.inputs.items()
            )
            args_doc = f"Args:\n{textwrap.indent(args_descriptions, '    ')}"
            tool_doc += f"\n\n{args_doc}"

        # Add returns documentation with output schema if it exists
        if has_schema:
            formatted_schema = json.dumps(self.output_schema, indent=4)
            indented_schema = textwrap.indent(formatted_schema, "        ")
            returns_doc = f"\nReturns:\n    dict (structured output): This tool ALWAYS returns a dictionary that strictly adheres to the following JSON schema:\n{indented_schema}"
            tool_doc += f"\n{returns_doc}"

        tool_doc = f'"""{tool_doc}\n"""'
        return f"def {self.name}{tool_signature}:\n{textwrap.indent(tool_doc, '    ')}"

    def to_tool_calling_prompt(self) -> str:
        return f"{self.name}: {self.description}\n    Takes inputs: {self.inputs}\n    Returns an output of type: {self.output_type}"

    def to_dict(self) -> dict:
        """Returns a dictionary representing the tool.

        Note: Inherit from Smolagents impl.
        """
        class_name = self.__class__.__name__
        if type(self).__name__ == "SimpleTool":
            # Check that imports are self-contained
            source_code = get_source(self.forward).replace("@tool", "")
            forward_node = ast.parse(source_code)
            # If tool was created using '@tool' decorator,
            # it has only a forward pass, so it's simpler to just get its code
            method_checker = MethodChecker(set())
            method_checker.visit(forward_node)

            if len(method_checker.errors) > 0:
                errors = [f"- {error}" for error in method_checker.errors]
                raise (
                    ValueError(
                        f"SimpleTool validation failed for {self.name}:\n"
                        + "\n".join(errors)
                    )
                )

            forward_source_code = get_source(self.forward)
            tool_code = textwrap.dedent(
                f"""
            from quantmind import Tool
            from typing import Any, Optional

            class {class_name}(Tool):
                name = "{self.name}"
                description = {json.dumps(textwrap.dedent(self.description).strip())}
                inputs = {repr(self.inputs)}
                output_type = "{self.output_type}"
            """
            ).strip()

            # Add output_schema if it exists
            if (
                hasattr(self, "output_schema")
                and self.output_schema is not None
            ):
                tool_code += f"\n                output_schema = {repr(self.output_schema)}"
            import re

            def add_self_argument(source_code: str) -> str:
                """Add 'self' as first argument to a function definition if not present."""
                pattern = r"def forward\(((?!self)[^)]*)\)"

                def replacement(match):
                    args = match.group(1).strip()
                    if args:  # If there are other arguments
                        return f"def forward(self, {args})"
                    return "def forward(self)"

                return re.sub(pattern, replacement, source_code)

            forward_source_code = forward_source_code.replace(
                self.name, "forward"
            )
            forward_source_code = add_self_argument(forward_source_code)
            forward_source_code = forward_source_code.replace(
                "@tool", ""
            ).strip()
            tool_code += "\n\n" + textwrap.indent(forward_source_code, "    ")

        else:  # If the tool was not created by the @tool decorator, it was made by subclassing Tool
            validate_tool_attributes(self.__class__)

            tool_code = (
                "from typing import Any, Optional\n"
                + instance_to_source(self, base_cls=Tool)
            )

        requirements = {
            el
            for el in get_imports(tool_code)
            if el not in sys.stdlib_module_names
        } | {"quantmind"}

        tool_dict = {
            "name": self.name,
            "code": tool_code,
            "requirements": sorted(requirements),
        }

        # Add output_schema if it exists
        if hasattr(self, "output_schema") and self.output_schema is not None:
            tool_dict["output_schema"] = self.output_schema

        return tool_dict

    @classmethod
    def from_dict(cls, tool_dict: dict[str, Any], **kwargs) -> "Tool":
        """Create tool from a dictionary representation.

        Args:
            tool_dict (`dict[str, Any]`): Dictionary representation of the tool.
            **kwargs: Additional keyword arguments to pass to the tool's constructor.

        Returns:
            `Tool`: Tool object.
        """
        if "code" not in tool_dict:
            raise ValueError(
                "Tool dictionary must contain 'code' key with the tool source code"
            )

        tool = cls.from_code(tool_dict["code"], **kwargs)

        # Set output_schema if it exists in the dictionary
        if "output_schema" in tool_dict:
            tool.output_schema = tool_dict["output_schema"]

        return tool

    def save(self, output_dir: str | Path, tool_file_name: str = "tool"):
        """Saves the relevant code files for your tool.

        This will copy the code of your tool in `output_dir` as well as autogenerate:

        - a `{tool_file_name}.py` file containing the logic for your tool.

        Args:
            output_dir (`str` or `Path`): The folder in which you want to save your tool.
            tool_file_name (`str`, *optional*): The file name in which you want to save your tool.
        """
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save tool file
        self._write_file(
            output_path / f"{tool_file_name}.py", self._get_tool_code()
        )

    def _write_file(self, file_path: Path, content: str) -> None:
        """Writes content to a file with UTF-8 encoding."""
        file_path.write_text(content, encoding="utf-8")

    def _get_tool_code(self) -> str:
        """Get the tool's code."""
        return self.to_dict()["code"]

    def _get_requirements(self) -> str:
        """Get the requirements."""
        return "\n".join(self.to_dict()["requirements"])

    @classmethod
    def from_code(cls, tool_code: str, **kwargs):
        module = types.ModuleType("dynamic_tool")

        exec(tool_code, module.__dict__)

        # Find the Tool subclass
        tool_class = next(
            (
                obj
                for _, obj in inspect.getmembers(module, inspect.isclass)
                if issubclass(obj, Tool) and obj is not Tool
            ),
            None,
        )

        if tool_class is None:
            raise ValueError("No Tool subclass found in the code.")

        # Convert inputs from string representation to dictionary if needed
        # When tool code is serialized/deserialized, complex data structures like
        # dictionaries may be stored as string literals (e.g., "{'key': 'value'}")
        # ast.literal_eval safely evaluates these string literals back to Python objects
        if not isinstance(tool_class.inputs, dict):
            tool_class.inputs = ast.literal_eval(tool_class.inputs)

        # Handle output_schema if it exists and is a string representation
        # Similar to inputs, output_schema might be serialized as a string literal
        # and needs to be converted back to its original Python data structure
        if hasattr(tool_class, "output_schema") and isinstance(
            tool_class.output_schema, str
        ):
            # ast.literal_eval is safer than eval() as it only evaluates literals
            # (strings, numbers, tuples, lists, dicts, booleans, None)
            # and prevents execution of arbitrary code
            tool_class.output_schema = ast.literal_eval(
                tool_class.output_schema
            )

        return tool_class(**kwargs)


def add_description(description):
    """A decorator that adds a description to a function."""

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


def tool(tool_function: Callable) -> Tool:
    """Convert a function into an instance of a dynamically created Tool subclass.

    Args:
        tool_function (`Callable`): Function to convert into a Tool subclass.
            Should have type hints for each input and a type hint for the output.
            Should also have a docstring including the description of the function
            and an 'Args:' part where each argument is described.
    """
    tool_json_schema = get_json_schema(tool_function)["function"]
    if "return" not in tool_json_schema:
        if len(tool_json_schema["parameters"]["properties"]) == 0:
            tool_json_schema["return"] = {"type": "null"}
        else:
            raise TypeHintParsingException(
                "Tool return type not found: make sure your function has a return type hint!"
            )

    class SimpleTool(Tool):
        def __init__(self):
            self.is_initialized = True

    # Set the class attributes
    SimpleTool.name = tool_json_schema["name"]
    SimpleTool.description = tool_json_schema["description"]
    SimpleTool.inputs = tool_json_schema["parameters"]["properties"]
    SimpleTool.output_type = tool_json_schema["return"]["type"]

    # Set output_schema if it exists in the JSON schema
    if "output_schema" in tool_json_schema:
        SimpleTool.output_schema = tool_json_schema["output_schema"]
    elif (
        "return" in tool_json_schema and "schema" in tool_json_schema["return"]
    ):
        SimpleTool.output_schema = tool_json_schema["return"]["schema"]

    @wraps(tool_function)
    def wrapped_function(*args, **kwargs):
        return tool_function(*args, **kwargs)

    # Bind the copied function to the forward method
    SimpleTool.forward = staticmethod(wrapped_function)

    # Get the signature parameters of the tool function
    sig = inspect.signature(tool_function)
    # - Add "self" as first parameter to tool_function signature
    new_sig = sig.replace(
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        + list(sig.parameters.values())
    )
    # - Set the signature of the forward method
    SimpleTool.forward.__signature__ = new_sig

    # Create and attach the source code of the dynamically created tool class and forward method
    # - Get the source code of tool_function
    tool_source = textwrap.dedent(inspect.getsource(tool_function))
    # - Remove the tool decorator and function definition line
    lines = tool_source.splitlines()
    tree = ast.parse(tool_source)
    #   - Find function definition
    func_node = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)),
        None,
    )
    if not func_node:
        raise ValueError(
            f"No function definition found in the provided source of {tool_function.__name__}. "
            "Ensure the input is a standard function."
        )
    #   - Extract decorator lines
    decorator_lines = ""
    if func_node.decorator_list:
        tool_decorators = [
            d
            for d in func_node.decorator_list
            if isinstance(d, ast.Name) and d.id == "tool"
        ]
        if len(tool_decorators) > 1:
            raise ValueError(
                f"Multiple @tool decorators found on function '{func_node.name}'. Only one @tool decorator is allowed."
            )
        if len(tool_decorators) < len(func_node.decorator_list):
            warnings.warn(
                f"Function '{func_node.name}' has decorators other than @tool. "
                "This may cause issues with serialization in the remote executor. See issue #1626."
            )
        decorator_start = (
            tool_decorators[0].end_lineno if tool_decorators else 0
        )
        decorator_end = func_node.decorator_list[-1].end_lineno
        decorator_lines = "\n".join(lines[decorator_start:decorator_end])
    #   - Extract tool source body
    body_start = func_node.body[0].lineno - 1  # AST lineno starts at 1
    tool_source_body = "\n".join(lines[body_start:])
    # - Create the forward method source, including def line and indentation
    forward_method_source = f"def forward{new_sig}:\n{tool_source_body}"
    # - Create the class source
    indent = " " * 4  # for class method
    class_source = (
        textwrap.dedent(f"""
        class SimpleTool(Tool):
            name: str = "{tool_json_schema["name"]}"
            description: str = {json.dumps(textwrap.dedent(tool_json_schema["description"]).strip())}
            inputs: dict[str, dict[str, str]] = {tool_json_schema["parameters"]["properties"]}
            output_type: str = "{tool_json_schema["return"]["type"]}"

            def __init__(self):
                self.is_initialized = True

        """)
        + textwrap.indent(decorator_lines, indent)
        + textwrap.indent(forward_method_source, indent)
    )
    # - Store the source code on both class and method for inspection
    SimpleTool.__source__ = class_source
    SimpleTool.forward.__source__ = forward_method_source

    simple_tool = SimpleTool()
    return simple_tool


def get_tools_definition_code(tools: dict[str, Tool]) -> str:
    """Get the tools definition code.

    This function gets the tools definition code.

    Args:
        tools (`dict[str, Tool]`): The tools to get the definition code for.

    Returns:
        `str`: The tools definition code.
    """
    tool_codes = []
    for tool in tools.values():
        validate_tool_attributes(tool.__class__, check_imports=False)
        tool_code = instance_to_source(tool, base_cls=Tool)
        tool_code = tool_code.replace("from quantmind.tools import Tool", "")
        tool_code += f"\n\n{tool.name} = {tool.__class__.__name__}()\n"
        tool_codes.append(tool_code)

    tool_definition_code = "\n".join(
        [f"import {module}" for module in BASE_BUILTIN_MODULES]
    )
    tool_definition_code += textwrap.dedent(
        """
    from typing import Any

    class Tool:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, *args, **kwargs):
            pass # to be implemented in child class
    """
    )
    tool_definition_code += "\n\n".join(tool_codes)
    return tool_definition_code


def validate_tool_arguments(tool: Tool, arguments: Any) -> None:
    """Validate tool arguments against tool's input schema.

    Checks that all provided arguments match the tool's expected input types and that
    all required arguments are present. Supports both dictionary arguments and single
    value arguments for tools with one input parameter.

    Args:
        tool (`Tool`): Tool whose input schema will be used for validation.
        arguments (`Any`): Arguments to validate. Can be a dictionary mapping
            argument names to values, or a single value for tools with one input.


    Raises:
        ValueError: If an argument is not in the tool's input schema, if a required
            argument is missing, or if the argument value doesn't match the expected type.
        TypeError: If an argument has an incorrect type that cannot be converted
            (e.g., string instead of number, excluding integer to number conversion).

    Note:
        - Supports type coercion from integer to number
        - Handles nullable parameters when explicitly marked in the schema
        - Accepts "any" type as a wildcard that matches all types
    """
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            if key not in tool.inputs:
                raise ValueError(
                    f"Argument {key} is not in the tool's input schema"
                )

            actual_type = _get_json_schema_type(type(value))["type"]
            expected_type = tool.inputs[key]["type"]
            expected_type_is_nullable = tool.inputs[key].get("nullable", False)

            # Type is valid if it matches, is "any", or is null for nullable parameters
            if (
                (
                    actual_type != expected_type
                    if isinstance(expected_type, str)
                    else actual_type not in expected_type
                )
                and expected_type != "any"
                and not (actual_type == "null" and expected_type_is_nullable)
            ):
                if actual_type == "integer" and expected_type == "number":
                    continue
                raise TypeError(
                    f"Argument {key} has type '{actual_type}' but should be '{tool.inputs[key]['type']}'"
                )

        for key, schema in tool.inputs.items():
            key_is_nullable = schema.get("nullable", False)
            if key not in arguments and not key_is_nullable:
                raise ValueError(f"Argument {key} is required")
        return None
    else:
        expected_type = list(tool.inputs.values())[0]["type"]
        if (
            _get_json_schema_type(type(arguments))["type"] != expected_type
            and not expected_type == "any"
        ):
            raise TypeError(
                f"Argument has type '{type(arguments).__name__}' but should be '{expected_type}'"
            )


__all__ = [
    "AUTHORIZED_TYPES",
    "Tool",
    "tool",
]
