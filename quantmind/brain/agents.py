#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import textwrap
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Type

from jinja2 import StrictUndefined, Template
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

if TYPE_CHECKING:
    import PIL.Image

from quantmind.models.agent import (
    DEFAULT_TOOLCALLING_PROMPTS,
    EMPTY_PROMPT_TEMPLATES,
    ActionOutput,
    PromptTemplates,
    RunResult,
    ToolOutput,
)
from quantmind.models.memory import (
    ActionStep,
    FinalAnswerStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    TokenUsage,
    ToolCall,
)
from quantmind.models.messages import (
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    agglomerate_stream_deltas,
    parse_json_if_needed,
)
from quantmind.tools import BaseTool, Tool, validate_tool_arguments
from quantmind.tools.default_tools import TOOL_MAPPING, FinalAnswerTool
from quantmind.utils.agentic_ext import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    is_valid_name,
    truncate_content,
)
from quantmind.utils.logger import get_logger
from quantmind.utils.monitoring import (
    YELLOW_HEX,
    AgentLogger,
    LogLevel,
    Monitor,
)

from .memory import CallbackRegistry, Memory

logger = get_logger(__name__)


def populate_template(template: str, variables: dict[str, Any]) -> str:
    """Populate a Jinja2 template with the given variables.

    Args:
        template (str): The Jinja2 template string to populate.
        variables (dict[str, Any]): Dictionary of variables to substitute in the template.

    Returns:
        str: The rendered template string with variables substituted.

    Raises:
        Exception: If there's an error during Jinja2 template rendering, including
            undefined variables or syntax errors.
    """
    compiled_template = Template(template, undefined=StrictUndefined)
    try:
        return compiled_template.render(**variables)
    except Exception as e:
        raise Exception(
            f"Error during jinja template rendering: {type(e).__name__}: {e}"
        )


class MultiStepAgent(ABC):
    """Multi-step agent class.

    Agent class that solves the given task step by step, using the ReAct framework:
    While the objective is not reached, the agent will perform a cycle of action (given by
    the LLM) and observation (obtained from the environment).

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Callable[[list[dict[str, str]]], ChatMessage]`): Model that will generate
            the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        instructions (`str`, *optional*): Custom instructions for the agent, will be
            inserted in the system prompt.
        max_steps (`int`, default `20`): Maximum number of steps the agent can take to
            solve the task.
        add_base_tools (`bool`, default `False`): Whether to add the base tools to the
            agent's tools.
        verbosity_level (`LogLevel`, default `LogLevel.INFO`): Level of verbosity of the
            agent's logs.
        managed_agents (`list`, *optional*): Managed agents that the agent can call.
        step_callbacks (`list[Callable]` | `dict[Type[MemoryStep], Callable |
            list[Callable]]`, *optional*): Callbacks that will be called at each step.
        planning_interval (`int`, *optional*): Interval at which the agent will run a
            planning step.
        name (`str`, *optional*): Necessary for a managed agent only - the name by which
            this agent can be called.
        description (`str`, *optional*): Necessary for a managed agent only - the
            description of this agent.
        provide_run_summary (`bool`, *optional*): Whether to provide a run summary when
            called as a managed agent.
        final_answer_checks (`list[Callable]`, *optional*): List of validation functions
            to run before accepting a final answer. Each function should:
            - Take the final answer and the agent's memory as arguments.
            - Return a boolean indicating whether the final answer is valid.
        return_full_result (`bool`, default `False`): Whether to return the full
            [`RunResult`] object or just the final answer output from the agent run.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Any,
        prompt_templates: PromptTemplates | None = None,
        instructions: str | None = None,
        max_steps: int = 20,
        add_base_tools: bool = False,
        verbosity_level: LogLevel = LogLevel.INFO,
        managed_agents: list | None = None,
        step_callbacks: list[Callable]
        | dict[Type[MemoryStep], Callable | list[Callable]]
        | None = None,
        planning_interval: int | None = None,
        name: str | None = None,
        description: str | None = None,
        provide_run_summary: bool = False,
        final_answer_checks: list[Callable] | None = None,
        return_full_result: bool = False,
        logger: AgentLogger | None = None,
    ):
        self.agent_name = self.__class__.__name__
        self.model = model
        self.prompt_templates = prompt_templates or EMPTY_PROMPT_TEMPLATES
        if prompt_templates is not None:
            missing_keys = set(EMPTY_PROMPT_TEMPLATES.keys()) - set(
                prompt_templates.keys()
            )
            assert not missing_keys, f"Some prompt templates are missing from your custom `prompt_templates`: {missing_keys}"
            for key, value in EMPTY_PROMPT_TEMPLATES.items():
                if isinstance(value, dict):
                    for subkey in value.keys():
                        assert (
                            key in prompt_templates.keys()
                            and (subkey in prompt_templates[key].keys())
                        ), f"Some prompt templates are missing from your custom `prompt_templates`: {subkey} under {key}"

        self.max_steps = max_steps
        self.step_number = 0
        self.planning_interval = planning_interval
        # TODO: More standardized state data structure?
        self.state: dict[str, Any] = {}
        self.name = self._validate_name(name)
        self.description = description
        self.provide_run_summary = provide_run_summary
        self.final_answer_checks = (
            final_answer_checks if final_answer_checks is not None else []
        )
        self.return_full_result = return_full_result
        self.instructions = instructions
        self._setup_managed_agents(managed_agents)
        self._setup_tools(tools, add_base_tools)
        self._validate_tools_and_managed_agents(tools, managed_agents)

        self.task: str | None = None
        self.memory = Memory(self.system_prompt)

        if logger is None:
            self.logger = AgentLogger(level=verbosity_level)
        else:
            self.logger = logger

        self.monitor = Monitor(self.model, self.logger)
        self._setup_step_callbacks(step_callbacks)
        self.stream_outputs = False

    @property
    def system_prompt(self) -> str:
        return self.initialize_system_prompt()

    @system_prompt.setter
    def system_prompt(self, value: str):
        raise AttributeError(
            """The 'system_prompt' property is read-only. Use 'self.prompt_templates["system_prompt"]' instead."""
        )

    def _validate_name(self, name: str | None) -> str | None:
        if name is not None and not is_valid_name(name):
            raise ValueError(
                f"Agent name '{name}' must be a valid Python identifier and not a reserved keyword."
            )
        return name

    def _setup_managed_agents(self, managed_agents: list | None = None) -> None:
        """Setup managed agents with proper logging."""
        self.managed_agents = {}
        if managed_agents:
            assert all(
                agent.name and agent.description for agent in managed_agents
            ), "All managed agents need both a name and a description!"
            self.managed_agents = {
                agent.name: agent for agent in managed_agents
            }
            # Ensure managed agents can be called as tools by the model:
            # set their inputs and output_type
            for agent in self.managed_agents.values():
                agent.inputs = {
                    "task": {
                        "type": "string",
                        "description": "Long detailed description of the task.",
                    },
                    "additional_args": {
                        "type": "object",
                        "description": "Dictionary of extra inputs to pass to the managed agent, e.g. images, dataframes, or any other contextual data it may need.",
                    },
                }
                agent.output_type = "string"

    def _setup_tools(self, tools: list[Tool], add_base_tools: bool) -> None:
        """Setup tools for the agent."""
        assert all(
            isinstance(tool, BaseTool) for tool in tools
        ), "All elements must be instance of BaseTool (or a subclass)"
        self.tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.tools.update(
                {
                    name: cls()
                    for name, cls in TOOL_MAPPING.items()
                    if name != "python_interpreter"
                    or self.__class__.__name__ == "ToolCallingAgent"
                }
            )
        self.tools.setdefault("final_answer", FinalAnswerTool())

    def _validate_tools_and_managed_agents(
        self, tools: list[Tool], managed_agents: list[Any] | None
    ) -> None:
        """Validate tools and managed agents."""
        tool_and_managed_agent_names = [tool.name for tool in tools]

        # Agent as tool
        if managed_agents is not None:
            tool_and_managed_agent_names += [
                agent.name for agent in managed_agents
            ]

        # TODO: Why self.name is added?
        if self.name:
            tool_and_managed_agent_names.append(self.name)
        if len(tool_and_managed_agent_names) != len(
            set(tool_and_managed_agent_names)
        ):
            raise ValueError(
                "Each tool or managed_agent should have a unique name! You passed these duplicate names: "
                f"{[name for name in tool_and_managed_agent_names if tool_and_managed_agent_names.count(name) > 1]}"
            )

    def _setup_step_callbacks(self, step_callbacks):
        # Initialize step callbacks registry
        self.step_callbacks = CallbackRegistry()
        if step_callbacks:
            # Register callbacks list only for ActionStep for backward compatibility
            if isinstance(step_callbacks, list):
                for callback in step_callbacks:
                    self.step_callbacks.register(ActionStep, callback)
            # Register callbacks dict for specific step classes
            elif isinstance(step_callbacks, dict):
                for step_cls, callbacks in step_callbacks.items():
                    if not isinstance(callbacks, list):
                        callbacks = [callbacks]
                    for callback in callbacks:
                        self.step_callbacks.register(step_cls, callback)
            else:
                raise ValueError("step_callbacks must be a list or a dict")
        # Register monitor update_metrics only for ActionStep for backward compatibility
        self.step_callbacks.register(ActionStep, self.monitor.update_metrics)

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
        return_full_result: bool | None = None,
    ) -> Any | RunResult:
        """Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode. If `True`, returns a
                generator that yields each step as it is executed. You must iterate
                over this generator to process the individual steps (e.g., using a
                for loop or `next()`). If `False`, executes all steps internally and
                returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from
                previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want
                to pass to the agent run, for instance images or dataframes. Give
                them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take
                to solve the task. if not provided, will use the agent's default
                value.
            return_full_result (`bool`, *optional*): Whether to return the full
                [`RunResult`] object or just the final answer output. If `None`
                (default), the agent's `self.return_full_result` setting is used.

        Example:
        ```py
        from quantmind.brain.agents import ToolCallingAgent

        agent = ToolCallingAgent(tools=[], model=my_model)
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access directly using the keys as variables:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(
            system_prompt=self.system_prompt
        )
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools(
                {**self.tools, **self.managed_agents}
            )

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run_stream(
                task=self.task, max_steps=max_steps, images=images
            )

        run_start_time = time.time()
        steps = list(
            self._run_stream(task=self.task, max_steps=max_steps, images=images)
        )

        # Outputs are returned only at the end. We only look at the last step.
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        return_full_result = (
            return_full_result
            if return_full_result is not None
            else self.return_full_result
        )
        if return_full_result:
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                )
            else:
                token_usage = None

            if self.memory.steps and isinstance(
                getattr(self.memory.steps[-1], "error", None),
                AgentMaxStepsError,
            ):
                state = "max_steps_error"
            else:
                state = "success"

            step_dicts = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                steps=step_dicts,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output

    def _run_stream(
        self,
        task: str,
        max_steps: int,
        images: list["PIL.Image.Image"] | None = None,
    ) -> Generator[
        ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta
    ]:
        self.step_number = 1
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= max_steps:
            if self.interrupt_switch:
                raise AgentError("Agent interrupted.", self.logger)

            # Run a planning step if scheduled
            if self.planning_interval is not None and (
                self.step_number == 1
                or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_start_time = time.time()
                planning_step = None
                for element in self._generate_planning_step(
                    task,
                    is_first_step=len(self.memory.steps) == 1,
                    step=self.step_number,
                ):  # Don't use the attribute step_number here, because there can be steps from previous runs
                    yield element
                    planning_step = element
                assert isinstance(
                    planning_step, PlanningStep
                )  # Last yielded element should be a PlanningStep
                planning_end_time = time.time()
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )
                self._finalize_step(planning_step)
                self.memory.steps.append(planning_step)

            # Start action step!
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            self.logger.log_rule(
                f"Step {self.step_number}", level=LogLevel.INFO
            )
            try:
                for output in self._step_stream(action_step):
                    # Yield all
                    yield output

                    if (
                        isinstance(output, ActionOutput)
                        and output.is_final_answer
                    ):
                        final_answer = output.output
                        self.logger.log(
                            Text(
                                f"Final answer: {final_answer}",
                                style=f"bold {YELLOW_HEX}",
                            ),
                            level=LogLevel.INFO,
                        )

                        if self.final_answer_checks:
                            self._validate_final_answer(final_answer)
                        returned_final_answer = True
                        action_step.is_final_answer = True

            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation
                # error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if not returned_final_answer and self.step_number == max_steps + 1:
            final_answer = self._handle_max_steps_reached(task)
            yield action_step
        yield FinalAnswerStep(final_answer)

    def _validate_final_answer(self, final_answer: Any):
        for check_function in self.final_answer_checks:
            try:
                assert check_function(final_answer, self.memory)
            except Exception as e:
                raise AgentError(
                    f"Check {check_function.__name__} failed with error: {e}",
                    self.logger,
                )

    def _finalize_step(self, memory_step: ActionStep | PlanningStep):
        memory_step.timing.end_time = time.time()
        self.step_callbacks.callback(memory_step, agent=self)

    def _handle_max_steps_reached(self, task: str) -> Any:
        action_step_start_time = time.time()
        final_answer = self.provide_final_answer(task)
        final_memory_step = ActionStep(
            step_number=self.step_number,
            error=AgentMaxStepsError("Reached max steps.", self.logger),
            timing=Timing(
                start_time=action_step_start_time, end_time=time.time()
            ),
            token_usage=final_answer.token_usage,
        )
        final_memory_step.action_output = final_answer.content
        self._finalize_step(final_memory_step)
        self.memory.steps.append(final_memory_step)
        return final_answer.content

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        start_time = time.time()
        if is_first_step:
            input_messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"][
                                    "initial_plan"
                                ],
                                variables={
                                    "task": task,
                                    "tools": self.tools,
                                    "managed_agents": self.managed_agents,
                                },
                            ),
                        }
                    ],
                )
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(
                    input_messages, stop_sequences=["<end_plan>"]
                )  # type: ignore
                input_tokens, output_tokens = 0, 0
                with Live(
                    "", console=self.logger.console, vertical_overflow="visible"
                ) as live:
                    for event in output_stream:
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                input_tokens = event.token_usage.input_tokens
                                output_tokens += event.token_usage.output_tokens
                        yield event
            else:
                plan_message = self.model.generate(
                    input_messages, stop_sequences=["<end_plan>"]
                )
                plan_message_content = plan_message.content
                input_tokens, output_tokens = 0, 0
                if plan_message.token_usage:
                    input_tokens = plan_message.token_usage.input_tokens
                    output_tokens = plan_message.token_usage.output_tokens
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by
            # the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"][
                                "update_plan_pre_messages"
                            ],
                            variables={"task": task},
                        ),
                    }
                ],
            )
            plan_update_post = ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"][
                                "update_plan_post_messages"
                            ],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                            },
                        ),
                    }
                ],
            )
            input_messages = (
                [plan_update_pre] + memory_messages + [plan_update_post]
            )
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                with Live(
                    "", console=self.logger.console, vertical_overflow="visible"
                ) as live:
                    for event in self.model.generate_stream(
                        input_messages,
                        stop_sequences=["<end_plan>"],
                    ):  # type: ignore
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                input_tokens = event.token_usage.input_tokens
                                output_tokens += event.token_usage.output_tokens
                        yield event
            else:
                plan_message = self.model.generate(
                    input_messages, stop_sequences=["<end_plan>"]
                )
                plan_message_content = plan_message.content
                input_tokens, output_tokens = 0, 0
                if plan_message.token_usage:
                    input_tokens = plan_message.token_usage.input_tokens
                    output_tokens = plan_message.token_usage.output_tokens
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(
            Rule(f"[bold]{log_headline}", style="orange"),
            Text(plan),
            level=LogLevel.INFO,
        )
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(
                role=MessageRole.ASSISTANT, content=plan_message_content
            ),
            token_usage=TokenUsage(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    @abstractmethod
    def initialize_system_prompt(self) -> str:
        """To be implemented in child classes."""
        ...

    def interrupt(self):
        """Interrupts the agent execution."""
        self.interrupt_switch = True

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """Write memory to messages.

        Reads past llm_outputs, actions, and observations or errors from the memory into a series of
        messages that can be used as input to the LLM. Adds a number of keywords (such as PLAN,
        error, etc) to help the LLM.
        """
        messages = self.memory.system_prompt.to_messages(
            summary_mode=summary_mode
        )
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[
        ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput
    ]:
        """Perform one step in the ReAct framework.

        The agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        raise NotImplementedError(
            "This method should be implemented in child classes"
        )

    def step(self, memory_step: ActionStep) -> Any:
        """Perform one step in the ReAct framework.

        The agent thinks, acts, and observes the result.
        Returns either None if the step is not final, or the final answer.
        """
        return list(self._step_stream(memory_step))[-1]

    def extract_action(
        self, model_output: str, split_token: str
    ) -> tuple[str, str]:
        """Parse action from the LLM output.

        Args:
            model_output (`str`): Output of the LLM
            split_token (`str`): Separator for the action. Should match the example in the system
                prompt.
        """
        try:
            split = model_output.split(split_token)
            rationale, action = (
                split[-2],
                split[-1],
            )  # NOTE: using indexes starting from the end solves for when you have more than one split_token in the output
        except Exception:
            raise AgentParsingError(
                f"No '{split_token}' token provided in your output.\nYour output:\n{model_output}\n. Be sure to include an action, prefaced with '{split_token}'!",
                self.logger,
            )
        return rationale.strip(), action.strip()

    def provide_final_answer(self, task: str) -> ChatMessage:
        """Provide the final answer to the task, based on the logs of the agent's interactions.

        Args:
            task (`str`): Task to perform.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.

        Returns:
            `str`: Final answer to the task.
        """
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": self.prompt_templates["final_answer"][
                            "pre_messages"
                        ],
                    }
                ],
            )
        ]
        messages += self.write_memory_to_messages()[1:]
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["final_answer"][
                                "post_messages"
                            ],
                            variables={"task": task},
                        ),
                    }
                ],
            )
        )
        try:
            chat_message: ChatMessage = self.model.generate(messages)
            return chat_message
        except Exception as e:
            return ChatMessage(
                role=MessageRole.ASSISTANT,
                content=[
                    {
                        "type": "text",
                        "text": f"Error in generating final LLM output: {e}",
                    }
                ],
            )

    def visualize(self):
        """Creates a rich tree visualization of the agent's structure."""
        self.logger.visualize_agent_tree(self)

    def replay(self, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to
                False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        self.memory.replay(self.logger, detailed=detailed)

    def __call__(self, task: str, **kwargs):
        """MultiStepAgent Call.

        Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task),
        )
        result = self.run(full_task, **kwargs)
        if isinstance(result, RunResult):
            report = result.output
        else:
            report = result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"],
            variables=dict(name=self.name, final_answer=report),
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message.content
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer


class ToolCallingAgent(MultiStepAgent):
    """Tool calling agent.

    This agent uses JSON-like tool calls, using method `model.get_tool_call` to
    leverage the LLM engine's tool calling capabilities.

    Args:
        tools: Tools that the agent can use.
        model: Model that will generate the agent's actions.
        prompt_templates: Prompt templates.
        planning_interval: Interval at which the agent will run a planning step.
        stream_outputs: Whether to stream outputs during execution.
        max_tool_threads: Maximum number of threads for parallel tool calls.
            Higher values increase concurrency but resource usage as well.
            Defaults to `ThreadPoolExecutor`'s default.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Any,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or DEFAULT_TOOLCALLING_PROMPTS
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # Tool calling setup
        self.max_tool_threads = max_tool_threads

    @property
    def tools_and_managed_agents(self):
        """Returns a combined list of tools and managed agents."""
        return list(self.tools.values()) + list(self.managed_agents.values())

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "custom_instructions": self.instructions,
            },
        )
        return system_prompt

    def _step_stream(
        self, memory_step: ActionStep
    ) -> Generator[
        ChatMessageStreamDelta | ToolCall | ToolOutput | ActionOutput
    ]:
        """Perform one step in the ReAct framework.

        The agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live(
                    "", console=self.logger.console, vertical_overflow="visible"
                ) as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(
                                agglomerate_stream_deltas(
                                    chat_message_stream_deltas
                                ).render_as_markdown()
                            )
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(
                    chat_message_stream_deltas
                )
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )
                if (
                    chat_message.content is None
                    and chat_message.raw is not None
                ):
                    log_content = str(chat_message.raw)
                else:
                    log_content = str(chat_message.content) or ""

                self.logger.log_markdown(
                    content=log_content,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # Record model output
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(
                f"Error while generating output:\n{e}", self.logger
            ) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(
                    f"Error while parsing tool call from model output: {e}",
                    self.logger,
                )
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(
                    tool_call.function.arguments
                )
        final_answer, got_final_answer = None, False
        for output in self.process_tool_calls(chat_message, memory_step):
            yield output
            if isinstance(output, ToolOutput):
                if output.is_final_answer:
                    if len(chat_message.tool_calls) > 1:
                        raise AgentExecutionError(
                            "If you want to return an answer, please do not perform any other tool calls than the final answer tool call!",
                            self.logger,
                        )
                    if got_final_answer:
                        raise AgentToolExecutionError(
                            "You returned multiple final answers. Please return only one single final answer!",
                            self.logger,
                        )
                    final_answer = output.output
                    got_final_answer = True

        yield ActionOutput(
            output=final_answer,
            is_final_answer=got_final_answer,
        )

    def process_tool_calls(
        self, chat_message: ChatMessage, memory_step: ActionStep
    ) -> Generator[ToolCall | ToolOutput]:
        """Process tool calls from the model output and update agent memory.

        Args:
            chat_message: Chat message containing tool calls from the model.
            memory_step: Memory ActionStep to update with results.

        Yields:
            The tool call or tool output.
        """
        parallel_calls: dict[str, ToolCall] = {}
        assert chat_message.tool_calls is not None
        for chat_tool_call in chat_message.tool_calls:
            tool_call = ToolCall(
                name=chat_tool_call.function.name,
                arguments=chat_tool_call.function.arguments,
                id=chat_tool_call.id,
            )
            yield tool_call
            parallel_calls[tool_call.id] = tool_call

        # Helper function to process a single tool call
        def process_single_tool_call(tool_call: ToolCall) -> ToolOutput:
            tool_name = tool_call.name
            tool_arguments = tool_call.arguments or {}
            self.logger.log(
                Panel(
                    Text(
                        f"Calling tool: '{tool_name}' with arguments: {tool_arguments}"
                    )
                ),
                level=LogLevel.INFO,
            )
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            observation = truncate_content(str(tool_call_result).strip())
            self.state[tool_call.id] = tool_call_result
            if observation == "":
                observation = "<empty>"
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            is_final_answer = tool_name == "final_answer"

            return ToolOutput(
                id=tool_call.id,
                output=tool_call_result,
                is_final_answer=is_final_answer,
                observation=observation,
                tool_call=tool_call,
            )

        # Process tool calls in parallel
        outputs = {}
        if len(parallel_calls) == 1:
            # If there's only one call, process it directly
            tool_call = list(parallel_calls.values())[0]
            tool_output = process_single_tool_call(tool_call)
            outputs[tool_output.id] = tool_output
            yield tool_output
        else:
            # If multiple tool calls, process them in parallel
            with ThreadPoolExecutor(self.max_tool_threads) as executor:
                futures = [
                    executor.submit(process_single_tool_call, tool_call)
                    for tool_call in parallel_calls.values()
                ]
                for future in as_completed(futures):
                    tool_output = future.result()
                    outputs[tool_output.id] = tool_output
                    yield tool_output

        memory_step.tool_calls = [
            parallel_calls[k] for k in sorted(parallel_calls.keys())
        ]
        memory_step.observations = memory_step.observations or ""
        for tool_output in [outputs[k] for k in sorted(outputs.keys())]:
            memory_step.observations += tool_output.observation + "\n"
        memory_step.observations = (
            memory_step.observations.rstrip("\n")
            if memory_step.observations
            else memory_step.observations
        )

    def _substitute_state_variables(
        self, arguments: dict[str, str] | str
    ) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value)
                if isinstance(value, str)
                else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(
        self, tool_name: str, arguments: dict[str, str] | str
    ) -> Any:
        """Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they
        refer to state variables.

        Args:
            tool_name: Name of the tool or managed agent to execute.
            arguments: Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.",
                self.logger,
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            validate_tool_arguments(tool, arguments)
        except (ValueError, TypeError) as e:
            raise AgentToolCallError(str(e), self.logger) from e
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}"
            raise AgentToolExecutionError(error_msg, self.logger) from e

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                if is_managed_agent:
                    return tool(**arguments)
                try:
                    return tool(**arguments, sanitize_inputs_outputs=True)
                except TypeError as type_error:
                    if "sanitize_inputs_outputs" not in str(type_error):
                        raise
                    return tool(**arguments)
            else:
                if is_managed_agent:
                    return tool(arguments)
                try:
                    return tool(arguments, sanitize_inputs_outputs=True)
                except TypeError as type_error:
                    if "sanitize_inputs_outputs" not in str(type_error):
                        raise
                    return tool(arguments)

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {str(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {str(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e
