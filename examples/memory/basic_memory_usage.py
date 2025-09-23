"""Minimal example showing how to build a memory timeline."""

from quantmind.brain.memory import Memory
from quantmind.models.memory import ActionStep, TaskStep, ToolCall
from quantmind.models.messages import ChatMessage, MessageRole
from quantmind.utils.monitoring import Timing, TokenUsage


def main():
    """Main function."""
    memory = Memory("You are a quantitative research assistant.")

    memory.steps.append(TaskStep(task="Gather the latest market sentiment."))

    memory.steps.append(
        ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=0.5),
            model_input_messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {"type": "text", "text": "Any updates on bond markets?"}
                    ],
                )
            ],
            tool_calls=[
                ToolCall(
                    name="fetch_sentiment",
                    arguments={"asset": "treasury", "lookback": "1d"},
                    id="call-1",
                )
            ],
            model_output="Sentiment looks neutral across regions.",
            observations="Tool returned neutral scores.",
            token_usage=TokenUsage(input_tokens=12, output_tokens=9),
        )
    )

    print("Succinct steps:")
    for step in memory.get_succinct_steps():
        print(step)

    print("\nMessages replay:")
    messages = []
    messages.extend(memory.system_prompt.to_messages())
    for step in memory.steps:
        messages.extend(step.to_messages())

    # Define colors for different message roles
    ROLE_COLORS = {
        MessageRole.SYSTEM: "\033[35m",  # Magenta
        MessageRole.USER: "\033[32m",  # Green
        MessageRole.ASSISTANT: "\033[36m",  # Cyan
        MessageRole.TOOL_CALL: "\033[33m",  # Yellow
        MessageRole.TOOL_RESPONSE: "\033[34m",  # Blue
    }
    RESET = "\033[0m"

    for message in messages:
        color = ROLE_COLORS.get(message.role, "")
        print(f"{color}[{message.role.value}]{RESET} {message.content}")


if __name__ == "__main__":
    main()
