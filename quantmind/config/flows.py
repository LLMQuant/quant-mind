"""Flow-specific configuration models for QuantMind with enhanced prompt engineering."""

import re
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from quantmind.config.llm import LLMConfig


# TODO(whisper): Need more simple flow config.
class BaseFlowConfig(BaseModel):
    """Base configuration for all flows with enhanced prompt engineering."""

    # LLM configuration - using composition pattern to avoid field duplication
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="LLM configuration"
    )

    # Enhanced Prompt Engineering Framework
    system_prompt: Optional[str] = Field(
        default=None, description="Per-instance system prompt override"
    )
    prompt_template: Optional[str] = Field(
        default=None,
        description="Custom prompt template with {{variable}} syntax",
    )
    template_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Custom template variables"
    )
    custom_instructions: Optional[str] = Field(
        default=None, description="Custom instructions to append to prompts"
    )

    # Custom prompt building function (advanced users)
    custom_build_prompt: Optional[Callable] = Field(
        default=None, description="Custom function for building prompts"
    )

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True  # Allow callable types

    @field_validator("prompt_template")
    def validate_template_syntax(cls, v):
        """Validate template uses correct {{variable}} syntax."""
        if v is None:
            return v

        # Check for balanced braces
        if v.count("{{") != v.count("}}"):
            raise ValueError(
                "Unbalanced template braces: use {{variable}} syntax"
            )

        return v

    # Convenience methods for creating configs with LLM parameters
    @classmethod
    def create(
        cls,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        custom_instructions: Optional[str] = None,
        **kwargs,
    ) -> "BaseFlowConfig":
        """Create a BaseFlowConfig with convenient LLM parameter specification.

        Args:
            model: LLM model name
            api_key: API key for LLM
            temperature: LLM temperature
            max_tokens: Maximum tokens
            custom_instructions: Custom instructions to append to prompts
            **kwargs: Additional flow-specific parameters

        Returns:
            Configured BaseFlowConfig instance
        """
        llm_config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_instructions=custom_instructions,
        )
        return cls(llm_config=llm_config, **kwargs)

    def get_system_prompt(self) -> str:
        """Get system prompt with fallback to class default."""
        # First check if LLM config has system prompt
        if self.llm_config.system_prompt:
            return self.llm_config.system_prompt
        # Then check flow-specific system prompt
        elif self.system_prompt:
            return self.system_prompt
        # Finally use class default
        else:
            return self.get_default_system_prompt()

    @classmethod
    def get_default_system_prompt(cls) -> str:
        """Override in subclasses for flow-specific defaults."""
        return (
            "You are an AI assistant specialized in quantitative finance analysis. "
            "Provide accurate, well-structured, and insightful responses based on the given content."
        )

    @classmethod
    def get_default_prompt_template(cls) -> str:
        """Override in subclasses for flow-specific default templates."""
        return (
            "{{system_prompt}}\n\n"
            "Task: Analyze the following content and provide insights.\n\n"
            "Title: {{title}}\n"
            "Abstract: {{abstract}}\n"
            "Content: {{content}}\n\n"
            "{{custom_instructions}}"
        )

    def extract_template_variables(
        self, knowledge_item, **kwargs
    ) -> Dict[str, Any]:
        """Extract variables for template substitution."""
        variables = {
            # Core knowledge item fields
            "title": knowledge_item.title or "",
            "abstract": knowledge_item.abstract or "",
            "content": knowledge_item.content or "",
            "authors": ", ".join(knowledge_item.authors)
            if knowledge_item.authors
            else "",
            "categories": ", ".join(knowledge_item.categories)
            if knowledge_item.categories
            else "",
            "tags": ", ".join(knowledge_item.tags)
            if knowledge_item.tags
            else "",
            "source": knowledge_item.source or "",
            "content_type": getattr(knowledge_item, "content_type", "generic"),
            # System components
            "system_prompt": self.get_system_prompt(),
            "custom_instructions": self.llm_config.custom_instructions or "",
            # Custom variables from config
            **self.template_variables,
            # Runtime variables
            **kwargs,
        }

        # Add meta_info with dot notation support
        if hasattr(knowledge_item, "meta_info") and knowledge_item.meta_info:
            for key, value in knowledge_item.meta_info.items():
                variables[f"meta_info.{key}"] = str(value)

        return variables

    def substitute_template(
        self, template: str, variables: Dict[str, Any]
    ) -> str:
        """Substitute {{variable}} syntax in template with user-friendly error handling."""

        def replace_var(match):
            var_name = match.group(1)
            if var_name in variables:
                return str(variables[var_name])
            else:
                # Graceful handling of missing variables
                return f"[{var_name}: not available]"

        # Replace {{variable}} with actual values
        result = re.sub(r"\{\{([^}]+)\}\}", replace_var, template)
        return result


class QAFlowConfig(BaseFlowConfig):
    """Configuration for Q&A generation flow."""

    # Q&A generation settings
    num_questions: int = Field(
        default=5, ge=1, description="Number of questions to generate"
    )
    include_different_difficulties: bool = Field(
        default=True,
        description="Generate questions with different difficulty levels",
    )
    focus_on_insights: bool = Field(
        default=True, description="Focus on generating insightful questions"
    )

    # Question categories
    question_categories: List[str] = Field(
        default_factory=lambda: [
            "basic_understanding",
            "methodology",
            "technical_details",
            "critical_analysis",
            "future_directions",
        ],
        description="Categories of questions to generate",
    )

    # Difficulty levels
    difficulty_levels: List[str] = Field(
        default_factory=lambda: [
            "beginner",
            "intermediate",
            "advanced",
            "expert",
        ],
        description="Difficulty levels for questions",
    )

    @classmethod
    def create(
        cls,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        num_questions: int = 5,
        custom_instructions: Optional[str] = None,
        **kwargs,
    ) -> "QAFlowConfig":
        """Create a QAFlowConfig with convenient LLM parameter specification."""
        from quantmind.config.llm import LLMConfig

        llm_config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_instructions=custom_instructions,
        )
        return cls(llm_config=llm_config, num_questions=num_questions, **kwargs)

    @classmethod
    def get_default_system_prompt(cls) -> str:
        """QA-specific system prompt."""
        return (
            "You are an expert in quantitative finance and financial research. "
            "Your task is to generate insightful questions and comprehensive answers "
            "based on academic papers and financial content. Focus on practical applications, "
            "methodology understanding, and critical analysis."
        )

    @classmethod
    def get_default_prompt_template(cls) -> str:
        """QA-specific prompt template."""
        return (
            "{{system_prompt}}\n\n"
            "Generate {{num_questions}} insightful questions and detailed answers based on this paper:\n\n"
            "Title: {{title}}\n"
            "Abstract: {{abstract}}\n"
            "Content: {{content}}\n\n"
            "Question Categories: {{question_categories}}\n"
            "Difficulty Levels: {{difficulty_levels}}\n\n"
            "{{custom_instructions}}\n\n"
            "Format your response as a structured JSON with questions and answers."
        )


class AnalyzerFlowConfig(BaseFlowConfig):
    """Configuration for content analysis flow."""

    # Tag analysis settings
    enable_tag_analysis: bool = Field(
        default=True, description="Enable tag analysis"
    )
    max_tags: int = Field(
        default=10, ge=1, description="Maximum number of tags to generate"
    )
    tag_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for tags",
    )

    # Analysis categories
    primary_categories: List[str] = Field(
        default_factory=lambda: [
            "market_type",
            "frequency",
            "methodology",
            "application",
        ],
        description="Primary tag categories",
    )

    secondary_categories: List[str] = Field(
        default_factory=lambda: [
            "data_source",
            "algorithm",
            "performance_metric",
            "risk_measure",
        ],
        description="Secondary tag categories",
    )

    # Summary generation
    generate_methodology_summary: bool = Field(
        default=True, description="Generate methodology summary"
    )
    generate_results_summary: bool = Field(
        default=True, description="Generate results summary"
    )
    summary_max_length: int = Field(
        default=500, ge=100, description="Maximum length for summaries"
    )

    @classmethod
    def create(
        cls,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        max_tags: int = 10,
        custom_instructions: Optional[str] = None,
        **kwargs,
    ) -> "AnalyzerFlowConfig":
        """Create an AnalyzerFlowConfig with convenient LLM parameter specification."""
        from quantmind.config.llm import LLMConfig

        llm_config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_instructions=custom_instructions,
        )
        return cls(llm_config=llm_config, max_tags=max_tags, **kwargs)

    @classmethod
    def get_default_system_prompt(cls) -> str:
        """Analyzer-specific system prompt."""
        return (
            "You are a financial content analyzer specialized in extracting structured insights "
            "from quantitative finance research. Focus on identifying key methodologies, "
            "market applications, data sources, and performance metrics."
        )

    @classmethod
    def get_default_prompt_template(cls) -> str:
        """Analyzer-specific prompt template."""
        return (
            "{{system_prompt}}\n\n"
            "Analyze the following content and extract structured information:\n\n"
            "Title: {{title}}\n"
            "Abstract: {{abstract}}\n"
            "Content: {{content}}\n\n"
            "Extract up to {{max_tags}} relevant tags from these categories:\n"
            "Primary: {{primary_categories}}\n"
            "Secondary: {{secondary_categories}}\n\n"
            "{{custom_instructions}}\n\n"
            "Provide analysis in structured JSON format."
        )


class SummaryFlowConfig(BaseFlowConfig):
    """Configuration for content summary generation flow."""

    # Summary settings
    summary_type: str = Field(
        default="comprehensive",
        description="Type of summary: 'brief', 'comprehensive', 'technical', 'executive'",
    )
    max_summary_length: int = Field(
        default=800, ge=100, description="Maximum length for summary"
    )
    include_key_findings: bool = Field(
        default=True, description="Include key findings in summary"
    )
    include_methodology: bool = Field(
        default=True, description="Include methodology overview"
    )
    include_results: bool = Field(
        default=True, description="Include main results"
    )
    include_implications: bool = Field(
        default=True, description="Include practical implications"
    )

    # Content focus
    focus_on_quantitative_aspects: bool = Field(
        default=True, description="Focus on quantitative and technical aspects"
    )
    highlight_innovations: bool = Field(
        default=True, description="Highlight innovative contributions"
    )
    include_limitations: bool = Field(
        default=True, description="Include limitations and caveats"
    )

    # Output format
    output_format: str = Field(
        default="structured",
        description="Output format: 'structured' (JSON), 'narrative' (text), 'bullet_points'",
    )
    include_confidence_score: bool = Field(
        default=False, description="Include confidence score in output"
    )

    @classmethod
    def create(
        cls,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        summary_type: str = "comprehensive",
        max_summary_length: int = 800,
        custom_instructions: Optional[str] = None,
        **kwargs,
    ) -> "SummaryFlowConfig":
        """Create a SummaryFlowConfig with convenient LLM parameter specification."""
        from quantmind.config.llm import LLMConfig

        llm_config = LLMConfig(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            custom_instructions=custom_instructions,
        )
        return cls(
            llm_config=llm_config,
            summary_type=summary_type,
            max_summary_length=max_summary_length,
            **kwargs,
        )

    @classmethod
    def get_default_system_prompt(cls) -> str:
        """Summary-specific system prompt."""
        return (
            "You are a financial research summarization expert. Create clear, comprehensive "
            "summaries that capture the essence of quantitative finance research papers. "
            "Focus on practical insights, methodological contributions, and actionable findings."
        )

    @classmethod
    def get_default_prompt_template(cls) -> str:
        """Summary-specific prompt template."""
        return (
            "{{system_prompt}}\n\n"
            "Create a {{summary_type}} summary (max {{max_summary_length}} words) for:\n\n"
            "Title: {{title}}\n"
            "Abstract: {{abstract}}\n"
            "Content: {{content}}\n\n"
            "Include: Key findings={{include_key_findings}}, "
            "Methodology={{include_methodology}}, Results={{include_results}}, "
            "Implications={{include_implications}}\n\n"
            "Format: {{output_format}}\n\n"
            "{{custom_instructions}}"
        )


class FlowConfig(BaseModel):
    """Flow configuration wrapper."""

    name: str
    type: str
    config: Union[BaseFlowConfig, Dict[str, Any]]
    enabled: bool = True
