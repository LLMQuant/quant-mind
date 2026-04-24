"""Default tools used by QuantMind agents."""

from .base import Tool


class FinalAnswerTool(Tool):
    """Simple tool that returns the final answer provided by the model."""

    name = "final_answer"
    description = (
        "Use this tool to provide the final answer or outcome of the task."
    )
    inputs = {
        "answer": {
            "type": "string",
            "description": "The final answer to return to the user.",
        }
    }
    output_type = "string"

    def forward(self, answer: str) -> str:
        return answer


# Mapping reserved for future default tools (e.g., search, python). Left empty
# so consumers can extend it without importing the original QuantMind defaults.
TOOL_MAPPING: dict[str, type[Tool]] = {}


__all__ = ["FinalAnswerTool", "TOOL_MAPPING"]
