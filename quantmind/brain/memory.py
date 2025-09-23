import inspect
from logging import getLogger
from typing import Callable, Type

from quantmind.models.memory import (
    ActionStep,
    MemoryStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
)
from quantmind.utils.monitoring import AgentLogger, LogLevel

logger = getLogger(__name__)


class Memory:
    """Memory for the brain, containing the system prompt and all steps taken by the brain.

    This class is used to store the agent's steps, including tasks, actions, and planning steps.
    It allows for resetting the memory, retrieving succinct or full step information, and replaying
    the agent's steps.

    Args:
        system_prompt (`str`): System prompt for the agent, which sets the context and instructions
            for the agent's behavior.

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- System prompt step for the agent.
        - **steps** (`list[TaskStep | ActionStep | PlanningStep]`) -- List of steps taken by the
            agent, which can include tasks, actions, and planning steps.
    """

    def __init__(self, system_prompt: str):
        self.system_prompt: SystemPromptStep = SystemPromptStep(
            system_prompt=system_prompt
        )
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self):
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """Return a succinct representation of the agent's steps, excluding model input messages."""
        return [
            {
                key: value
                for key, value in step.dict().items()
                if key != "model_input_messages"
            }
            for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """Return a full representation of the agent's steps, including model input messages."""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (`AgentLogger`): The logger to print replay logs to.
            detailed (`bool`, default `False`): If True, also displays the memory at each step.
                Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(
            title="System prompt",
            content=self.system_prompt.system_prompt,
            level=LogLevel.ERROR,
        )
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(
                    f"Step {step.step_number}", level=LogLevel.ERROR
                )
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(
                        step.model_input_messages, level=LogLevel.ERROR
                    )
                if step.model_output is not None:
                    logger.log_markdown(
                        title="Agent output:",
                        content=step.model_output,
                        level=LogLevel.ERROR,
                    )
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(
                        step.model_input_messages, level=LogLevel.ERROR
                    )
                logger.log_markdown(
                    title="Agent output:",
                    content=step.plan,
                    level=LogLevel.ERROR,
                )

    def return_full_code(self) -> str:
        """Returns all code actions from the agent's steps, concatenated as a single script."""
        return "\n\n".join(
            [
                step.code_action
                for step in self.steps
                if isinstance(step, ActionStep) and step.code_action is not None
            ]
        )


class CallbackRegistry:
    """Registry for callbacks that are called at each step of the agent's execution.

    Callbacks are registered by passing a step class and a callback function.
    """

    def __init__(self):
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}

    def register(self, step_cls: Type[MemoryStep], callback: Callable):
        """Register a callback for a step class.

        Args:
            step_cls (Type[MemoryStep]): Step class to register the callback for.
            callback (Callable): Callback function to register.
        """
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step, **kwargs):
        """Call callbacks registered for a step type.

        Args:
            memory_step (MemoryStep): Step to call the callbacks for.
            **kwargs: Additional arguments to pass to callbacks that accept them.
                Typically, includes the agent instance.

        Notes:
            For backwards compatibility, callbacks with a single parameter signature
            receive only the memory_step, while callbacks with multiple parameters
            receive both the memory_step and any additional kwargs.
        """
        # For compatibility with old callbacks that only take the step as an argument
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(memory_step) if len(
                    inspect.signature(cb).parameters
                ) == 1 else cb(memory_step, **kwargs)
