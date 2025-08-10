"""Base flow abstract class for QuantMind framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from jinja2 import Template

from quantmind.config import BaseFlowConfig, LLMConfig
from quantmind.llm import LLMBlock, create_llm_block
from quantmind.models.content import KnowledgeItem
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


# Type alias for flow input data
FlowInput = Union[
    KnowledgeItem, List[KnowledgeItem], Dict[str, KnowledgeItem], Any
]


class BaseFlow(ABC):
    """Abstract base class for all flows providing resource access and orchestration framework.

    BaseFlow provides:
    - Resource management (LLM blocks and prompt templates)
    - Helper methods for rendering prompts
    - Abstract run() method for subclass-specific business logic

    Flow subclasses implement the run() method with Python code to define
    the specific orchestration logic, conditions, loops, and parallel operations.
    """

    def __init__(self, config: BaseFlowConfig):
        """Initialize flow with configuration.

        Args:
            config: Flow configuration defining resources (LLM blocks and templates)
        """
        self.config = config
        self._llm_blocks = self._initialize_llm_blocks(config.llm_blocks)
        self._templates = {
            name: Template(template_str)
            for name, template_str in config.prompt_templates.items()
        }

        logger.info(
            f"Initialized flow '{config.name}' with {len(self._llm_blocks)} LLM blocks"
        )

    def _initialize_llm_blocks(
        self, llm_configs: Dict[str, LLMConfig]
    ) -> Dict[str, Union[LLMBlock, None]]:
        """Initialize LLM blocks from configurations.

        Args:
            llm_configs: Dictionary of LLM configurations

        Returns:
            Dictionary of initialized LLM blocks
        """
        llm_blocks = {}
        for identifier, llm_config in llm_configs.items():
            try:
                llm_block = create_llm_block(llm_config)
                llm_blocks[identifier] = llm_block
                logger.debug(
                    f"Initialized LLM block '{identifier}' with model: {llm_config.model}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to initialize LLM block '{identifier}': {e}"
                )
                llm_blocks[identifier] = None

        return llm_blocks

    def _render_prompt(self, template_name: str, **kwargs) -> str:
        """Render prompt using specified template and variables.

        Args:
            template_name: Name of template to use
            **kwargs: Template variables

        Returns:
            Rendered prompt string

        Raises:
            KeyError: If template not found
        """
        if template_name not in self._templates:
            raise KeyError(
                f"Template '{template_name}' not found in flow config"
            )

        template = self._templates[template_name]
        return template.render(**kwargs)

    @abstractmethod
    def run(self, flow_input: FlowInput) -> Any:
        """Execute the flow's business logic orchestration.

        This method must be implemented by subclasses to define the specific
        workflow steps, using the available LLM blocks and prompt templates.

        Args:
            flow_input: Initial input data (typically KnowledgeItem)

        Returns:
            Flow execution result (structure depends on flow)
        """
        pass
