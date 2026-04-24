import unittest
from typing import get_args, get_origin

from quantmind.models.agent import (
    DEFAULT_TOOLCALLING_PROMPTS,
    EMPTY_PROMPT_TEMPLATES,
    ActionOutput,
    RunResult,
    StreamEvent,
    ToolOutput,
)
from quantmind.models.memory import (
    ActionStep,
    FinalAnswerStep,
    PlanningStep,
    ToolCall,
)
from quantmind.models.messages import (
    ChatMessageStreamDelta,
    ChatMessageToolCall,
)
from quantmind.utils.monitoring import Timing, TokenUsage


class AgentModelTests(unittest.TestCase):
    """Tests for the dataclasses used by the self-built agent runtime."""

    def test_run_result_dict_and_messages_fallback(self) -> None:
        timing = Timing(start_time=1.0, end_time=2.0)
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        result = RunResult(
            output="ok",
            state="success",
            steps=[{"step": 1}],
            token_usage=usage,
            timing=timing,
        )
        as_dict = result.dict()
        self.assertEqual(as_dict["output"], "ok")
        self.assertEqual(as_dict["token_usage"]["total_tokens"], 15)
        # messages fallback path keeps backward compatibility
        with self.assertWarns(FutureWarning):
            legacy_result = RunResult(
                messages=[{"legacy": True}],
                timing=timing,
            )
        self.assertEqual(legacy_result.steps, [{"legacy": True}])

    def test_prompt_templates_structure(self) -> None:
        self.assertSetEqual(
            set(EMPTY_PROMPT_TEMPLATES.keys()),
            {"system_prompt", "planning", "managed_agent", "final_answer"},
        )
        self.assertIn("system_prompt", DEFAULT_TOOLCALLING_PROMPTS)
        self.assertIn("planning", DEFAULT_TOOLCALLING_PROMPTS)
        planning_prompts = DEFAULT_TOOLCALLING_PROMPTS["planning"]
        self.assertIn("initial_plan", planning_prompts)
        self.assertTrue(
            DEFAULT_TOOLCALLING_PROMPTS["system_prompt"]
        )  # non-empty default prompt

    def test_stream_event_alias_members(self) -> None:
        origin = get_origin(StreamEvent)
        self.assertIsNotNone(origin)
        args = set(get_args(StreamEvent))
        expected = {
            ChatMessageStreamDelta,
            ChatMessageToolCall,
            ActionOutput,
            ToolCall,
            ToolOutput,
            PlanningStep,
            ActionStep,
            FinalAnswerStep,
        }
        self.assertTrue(expected.issubset(args))


if __name__ == "__main__":
    unittest.main()
