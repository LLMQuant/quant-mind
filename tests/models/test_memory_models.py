import unittest

from quantmind.models.memory import (
    ActionStep,
    PlanningStep,
    TaskStep,
    ToolCall,
)
from quantmind.models.messages import ChatMessage, MessageRole
from quantmind.utils.monitoring import Timing, TokenUsage


class MemoryModelsTestCase(unittest.TestCase):
    """Test memory models functionality."""

    def test_action_step_dict_serializes_tool_calls_and_tokens(self):
        call = ToolCall(
            name="status_tool",
            arguments={"region": "APAC"},
            id="call-1",
        )
        step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=0.4),
            model_input_messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "Check signals"}],
                )
            ],
            tool_calls=[call],
            model_output="Signals retrieved",
            action_output={"status": "ok"},
            token_usage=TokenUsage(input_tokens=7, output_tokens=3),
        )

        data = step.dict()

        self.assertEqual(data["tool_calls"], [call.dict()])
        self.assertEqual(data["token_usage"]["total_tokens"], 10)
        self.assertEqual(data["model_output"], "Signals retrieved")
        self.assertIsNone(data["observations"])

    def test_action_step_to_messages_includes_output_and_observation(self):
        step = ActionStep(
            step_number=2,
            timing=Timing(start_time=1.0, end_time=1.5),
            model_output="Computation complete",
            observations="Done",
            tool_calls=[
                ToolCall(
                    name="compute",
                    arguments={"x": 1},
                    id="compute-1",
                )
            ],
        )

        messages = step.to_messages()

        self.assertEqual(messages[0].role, MessageRole.ASSISTANT)
        self.assertEqual(messages[0].content[0]["text"], "Computation complete")
        self.assertEqual(messages[1].role, MessageRole.TOOL_CALL)
        self.assertIn("Calling tools", messages[1].content[0]["text"])
        self.assertEqual(messages[-1].role, MessageRole.TOOL_RESPONSE)
        self.assertIn("Observation", messages[-1].content[0]["text"])

    def test_planning_step_to_messages(self):
        step = PlanningStep(
            model_input_messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "Plan it"}],
                )
            ],
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content="Plan follows",
            ),
            plan="Step 1",
            timing=Timing(start_time=2.0, end_time=2.5),
        )

        messages = step.to_messages()

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, MessageRole.ASSISTANT)
        self.assertEqual(messages[1].role, MessageRole.USER)

    def test_task_step_to_messages_handles_images(self):
        class _Image:
            def __init__(self, value: str):
                self._value = value

            def tobytes(self):
                return self._value

        fake_image = _Image("image-bytes")
        step = TaskStep(task="Summarize", task_images=[fake_image])
        messages = step.to_messages()

        self.assertEqual(messages[0].role, MessageRole.USER)
        self.assertEqual(messages[0].content[1]["type"], "image")
        self.assertIs(messages[0].content[1]["image"], fake_image)


if __name__ == "__main__":
    unittest.main()
