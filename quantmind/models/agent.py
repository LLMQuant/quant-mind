"""Agent-related data models and prompt templates."""

from __future__ import annotations

import textwrap
import warnings
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, Union

from quantmind.utils.monitoring import Timing, TokenUsage

from .memory import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    ToolCall,
)
from .messages import (
    ChatMessageStreamDelta,
    ChatMessageToolCall,
)


@dataclass
class ActionOutput:
    """Output of an action."""

    output: Any
    is_final_answer: bool


@dataclass
class ToolOutput:
    """Output of a tool."""

    id: str
    output: Any
    is_final_answer: bool
    observation: str
    tool_call: ToolCall


class PlanningPromptTemplate(TypedDict):
    """Prompt templates for the planning step."""

    initial_plan: str
    update_plan_pre_messages: str
    update_plan_post_messages: str


class ManagedAgentPromptTemplate(TypedDict):
    """Prompt templates for the managed agent."""

    task: str
    report: str


class FinalAnswerPromptTemplate(TypedDict):
    """Prompt templates for the final answer."""

    pre_messages: str
    post_messages: str


class PromptTemplates(TypedDict):
    """Prompt templates for the agent."""

    system_prompt: str
    planning: PlanningPromptTemplate
    managed_agent: ManagedAgentPromptTemplate
    final_answer: FinalAnswerPromptTemplate


EMPTY_PROMPT_TEMPLATES = PromptTemplates(
    system_prompt="",
    planning=PlanningPromptTemplate(
        initial_plan="",
        update_plan_pre_messages="",
        update_plan_post_messages="",
    ),
    managed_agent=ManagedAgentPromptTemplate(task="", report=""),
    final_answer=FinalAnswerPromptTemplate(pre_messages="", post_messages=""),
)


DEFAULT_TOOLCALLING_PROMPTS: PromptTemplates = {
    "system_prompt": textwrap.dedent(
        """
        You are an expert assistant that can call tools to solve the user's task. Think step-by-step.
        When you are finished, call the `final_answer` tool with the final response.
        """
    ).strip(),
    "planning": {
        "initial_plan": textwrap.dedent(
            """
            Summarize the task, list the key facts you know or need, then outline a concise plan.
            Finish your planning message with '<end_plan>'.
            """
        ).strip(),
        "update_plan_pre_messages": "Review the task and the recent history before updating the plan.",
        "update_plan_post_messages": textwrap.dedent(
            """
            Provide updated facts and an adjusted plan. End with '<end_plan>'.
            """
        ).strip(),
    },
    "managed_agent": {
        "task": "Act on the user's task using the available tools.",
        "report": "{{name}} completed the task with the following result: {{final_answer}}",
    },
    "final_answer": {
        "pre_messages": "Validate that you are ready to provide the final answer.",
        "post_messages": "Share the final answer clearly and concisely.",
    },
}


@dataclass
class RunResult:
    """Holds extended information about an agent run."""

    output: Any | None
    state: Literal["success", "max_steps_error"]
    steps: list[dict]
    token_usage: TokenUsage | None
    timing: Timing

    def __init__(
        self,
        output=None,
        state=None,
        steps=None,
        token_usage=None,
        timing=None,
        messages=None,
    ):
        if messages is not None:
            if steps is not None:
                raise ValueError(
                    "Cannot specify both 'messages' and 'steps' parameters. Use 'steps' instead."
                )
            warnings.warn(
                "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
                FutureWarning,
                stacklevel=2,
            )
            steps = messages

        self.output = output
        self.state = state
        self.steps = steps
        self.token_usage = token_usage
        self.timing = timing

    @property
    def messages(self):
        """Backward compatibility property that returns steps."""
        warnings.warn(
            "Parameter 'messages' is deprecated and will be removed in version 1.25. Please use 'steps' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.steps

    def dict(self):
        return {
            "output": self.output,
            "state": self.state,
            "steps": self.steps,
            "token_usage": self.token_usage.dict()
            if self.token_usage is not None
            else None,
            "timing": self.timing.dict(),
        }


StreamEvent = Union[
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ActionOutput,
    ToolCall,
    ToolOutput,
    PlanningStep,
    ActionStep,
    FinalAnswerStep,
]
