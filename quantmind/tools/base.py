from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, Type

from pydantic import BaseModel, Field, create_model


class BaseTool(ABC):
    """Abstract base class for all QuantMind tools.

    A tool self-describes its capability via a name, description, and a Pydantic
    input schema, and exposes an async `run` that validates inputs before execution.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used by an LLM to invoke this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the tool for LLM selection."""

    @property
    @abstractmethod
    def args_schema(self) -> Type[BaseModel]:
        """Pydantic model describing required/optional input arguments."""

    @abstractmethod
    async def _arun(self, **kwargs: Any) -> Any:
        """Core async execution logic for the tool (implemented by subclasses)."""

    async def run(self, **kwargs: Any) -> Any:
        """Validate inputs against schema, then execute the tool asynchronously."""
        validated = self.args_schema(**kwargs)
        return await self._arun(**validated.model_dump())

    def to_openai_schema(self) -> Dict[str, Any]:
        """Return schema compatible with OpenAI function calling tools."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


class FunctionTool(BaseTool):
    """Wrap a Python callable as a QuantMind tool.

    The callable may be sync or async. Sync functions are executed in a thread
    pool to avoid blocking the event loop.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        description: str,
        args_schema: Type[BaseModel],
    ) -> None:
        self._fn = fn
        self._name = name
        self._description = description
        self._args_schema = args_schema

    @property
    def name(self) -> str:  # type: ignore[override]
        return self._name

    @property
    def description(self) -> str:  # type: ignore[override]
        return self._description

    @property
    def args_schema(self) -> Type[BaseModel]:  # type: ignore[override]
        return self._args_schema

    async def _arun(self, **kwargs: Any) -> Any:  # type: ignore[override]
        if inspect.iscoroutinefunction(self._fn):
            return await self._fn(**kwargs)  # type: ignore[misc]
        # For sync functions, run in a thread pool
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, partial(self._fn, **kwargs))


def _build_args_schema_from_signature(
    fn: Callable[..., Any],
) -> Type[BaseModel]:
    """Create a Pydantic model from a function's signature.

    Parameters without annotations default to `Any`. All parameters are required
    unless a default value exists on the function.
    """
    sig = inspect.signature(fn)
    fields: Dict[str, tuple[Any, Any]] = {}

    for param in sig.parameters.values():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            # Skip variadic params for schema simplicity
            continue

        annotation = (
            param.annotation
            if param.annotation is not inspect.Parameter.empty
            else Any
        )

        # Required if no default, else use default
        if param.default is inspect.Parameter.empty:
            default = Field(..., description=f"Parameter for {param.name}")
        else:
            default = Field(
                default=param.default, description=f"Parameter for {param.name}"
            )

        fields[param.name] = (annotation, default)

    model_name = f"{fn.__name__.capitalize()}Inputs"
    return create_model(model_name, **fields)  # type: ignore[return-value]


def tool(fn: Callable[..., Any]) -> BaseTool:
    """Decorator that converts a function into a QuantMind Tool.

    The function's docstring becomes the description; its signature and type
    annotations define the input schema. Returns a `FunctionTool` instance.
    """
    docstring = inspect.getdoc(fn)
    if not docstring:
        raise ValueError(
            "Tool function must have a docstring for its description."
        )

    description = docstring.strip()
    name = fn.__name__
    args_schema = _build_args_schema_from_signature(fn)
    return FunctionTool(
        fn=fn, name=name, description=description, args_schema=args_schema
    )
