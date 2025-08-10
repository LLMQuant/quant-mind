"""Simple greeting flow demonstrating custom flow creation."""

from typing import Any, Dict

from quantmind.config.flows import BaseFlowConfig
from quantmind.config.llm import LLMConfig
from quantmind.config.registry import register_flow_config
from quantmind.flow.base import BaseFlow


@register_flow_config("greeting")
class GreetingFlowConfig(BaseFlowConfig):
    """Configuration for greeting flow."""

    def model_post_init(self, __context: Any) -> None:
        """Initialize default configuration."""
        # First load prompt templates from path if specified
        super().model_post_init(__context)

        if not self.llm_blocks:
            self.llm_blocks = {
                "greeter": LLMConfig(
                    model="gpt-4o-mini", temperature=0.7, max_tokens=500
                )
            }


class GreetingFlow(BaseFlow):
    """A simple custom flow that greets users and provides suggestions."""

    def run(self, user_input: Dict[str, Any]) -> Dict[str, str]:
        """Execute the greeting flow.

        Args:
            user_input: Dictionary containing 'user_name' and 'topic'

        Returns:
            Dictionary with greeting and suggestions
        """
        user_name = user_input.get("user_name", "there")
        topic = user_input.get("topic", "learning")

        # Step 1: Generate greeting
        greeter_llm = self._llm_blocks["greeter"]

        greeting_prompt = self._render_prompt(
            "greeting_template", user_name=user_name, topic=topic
        )

        greeting = greeter_llm.generate_text(greeting_prompt)

        # Step 2: Generate follow-up suggestions
        follow_up_prompt = self._render_prompt(
            "follow_up_template", user_name=user_name, topic=topic
        )

        suggestions = greeter_llm.generate_text(follow_up_prompt)

        return {
            "greeting": greeting or "Hello! Welcome to our system!",
            "suggestions": suggestions or "Keep exploring and learning!",
        }
