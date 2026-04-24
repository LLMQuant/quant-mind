import unittest

from quantmind.brain.agents import ToolCallingAgent
from quantmind.models.messages import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
)


class DummyModel:
    """Simple model stub that always requests the final answer tool."""

    def generate(self, *args, **kwargs) -> ChatMessage:
        tool_call = ChatMessageToolCall(
            function=ChatMessageToolCallFunction(
                name="final_answer",
                arguments={"answer": "42"},
                description=None,
            ),
            id="call_1",
            type="function",
        )
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=None,
            tool_calls=[tool_call],
        )

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        return message


class ToolCallingAgentTests(unittest.TestCase):
    """Smoke tests for the self-built ToolCallingAgent runtime."""

    def test_run_returns_final_answer(self) -> None:
        agent = ToolCallingAgent(tools=[], model=DummyModel())
        result = agent.run("Return 42")
        self.assertEqual(result, "42")


if __name__ == "__main__":
    unittest.main()
