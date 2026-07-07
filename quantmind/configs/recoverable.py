"""Recoverable parsing helpers for tolerant ingestion boundaries.

This module keeps strict schema contracts in place (`extra="forbid"` on
the target model) while preventing pipeline paralysis during malformed
inputs. Callers receive either a validated model or a structured
quarantine node that preserves raw context.
"""

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

T = TypeVar("T", bound=BaseModel)


class RawFallbackNode(BaseModel):
    """Quarantine payload used when schema validation fails."""

    model_config = ConfigDict(extra="forbid")

    raw_payload: dict[str, Any] = Field(default_factory=dict)
    target_schema: str
    error_message: str
    context_loss_prevented: bool = True


class RecoverableValidation(Generic[T]):
    """Validate against a target model and quarantine failures."""

    def __init__(self, target_model: type[T]) -> None:
        self.target_model = target_model

    def execute_safely(self, data: dict[str, Any]) -> T | RawFallbackNode:
        """Return validated model or a fallback node with original payload."""
        try:
            return self.target_model.model_validate(data)
        except ValidationError as error:
            return RawFallbackNode(
                raw_payload=data,
                target_schema=self.target_model.__name__,
                error_message=str(error),
            )
