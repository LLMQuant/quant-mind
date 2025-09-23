import unittest

from quantmind.brain.memory import CallbackRegistry, Memory
from quantmind.models.memory import ActionStep, TaskStep, ToolCall
from quantmind.models.messages import ChatMessage, MessageRole
from quantmind.utils.monitoring import Timing, TokenUsage


class MemoryTestCase(unittest.TestCase):
    """Test memory functionality."""

    def test_memory_succinct_and_full_steps(self):
        memory = Memory("system prompt")
        task_step = TaskStep(task="Investigate signal")
        action_step = ActionStep(
            step_number=1,
            timing=Timing(start_time=0.0, end_time=1.2),
            model_input_messages=[
                ChatMessage(
                    role=MessageRole.USER,
                    content=[{"type": "text", "text": "What is the status?"}],
                )
            ],
            tool_calls=[
                ToolCall(
                    name="status_tool",
                    arguments={"region": "EMEA", "threshold": 0.5},
                    id="call-1",
                )
            ],
            model_output="Status retrieved",
            action_output={"result": "ok"},
            token_usage=TokenUsage(input_tokens=10, output_tokens=5),
        )
        memory.steps.extend([task_step, action_step])

        succinct_steps = memory.get_succinct_steps()
        self.assertEqual(len(succinct_steps), 2)
        self.assertNotIn("model_input_messages", succinct_steps[1])

        full_steps = memory.get_full_steps()
        self.assertEqual(len(full_steps), 2)
        self.assertIn("model_input_messages", full_steps[1])
        self.assertEqual(
            full_steps[1]["tool_calls"][0]["function"]["name"],
            "status_tool",
        )
        self.assertEqual(full_steps[1]["token_usage"]["total_tokens"], 15)

    def test_action_step_to_messages(self):
        step = ActionStep(
            step_number=2,
            timing=Timing(start_time=5.0, end_time=6.0),
            model_output="Calculation complete",
            tool_calls=[
                ToolCall(
                    name="calc_tool",
                    arguments={"x": 1, "y": 2},
                    id="calc-1",
                )
            ],
            observations="All good",
        )

        messages = step.to_messages()
        self.assertEqual(messages[0].role, MessageRole.ASSISTANT)
        self.assertEqual(messages[0].content[0]["text"], "Calculation complete")
        self.assertEqual(messages[1].role, MessageRole.TOOL_CALL)
        self.assertIn("Calling tools", messages[1].content[0]["text"])
        self.assertEqual(messages[-1].role, MessageRole.TOOL_RESPONSE)
        self.assertIn("Observation", messages[-1].content[0]["text"])

    def test_callback_registry_executes_registered_callbacks(self):
        registry = CallbackRegistry()
        captured = []

        def on_action(step, agent=None):
            captured.append((step.step_number, agent))

        registry.register(ActionStep, on_action)

        dummy_step = ActionStep(
            step_number=3, timing=Timing(start_time=0.0, end_time=0.1)
        )
        registry.callback(dummy_step, agent="agent-instance")

        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], 3)
        self.assertEqual(captured[0][1], "agent-instance")


if __name__ == "__main__":
    unittest.main()
