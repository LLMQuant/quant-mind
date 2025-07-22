# QuantMind Flow Examples

This directory contains examples demonstrating the enhanced flow framework in QuantMind with advanced prompt engineering capabilities.

## Available Flows

### 1. Content Summary Flow (`content_summary_example.py`)

The `SummaryFlow` generates comprehensive summaries of knowledge items (research papers, articles, reports, etc.) using LLM technology with enhanced template-based prompt engineering.

**Features:**
- Multiple summary types: brief, comprehensive, technical, executive
- Flexible output formats: narrative, structured (JSON), bullet points
- Enhanced prompt engineering with {{variable}} template syntax
- Custom template variables and instructions support
- Generic KnowledgeItem support (not limited to papers)
- Backward compatibility with Paper objects
- Convenience methods for common summary types

**Usage:**
```python
from quantmind.flow import SummaryFlow
from quantmind.config import SummaryFlowConfig
from quantmind.models.content import KnowledgeItem

# Create configuration with custom template
config = SummaryFlowConfig(
    llm_type="openai",
    llm_name="gpt-4o",
    api_key="your-api-key",
    summary_type="comprehensive",
    max_summary_length=600,
    output_format="narrative",
    # Custom prompt template using {{variable}} syntax
    prompt_template=(
        "{{system_prompt}}\n\n"
        "Analyze this {{content_type}} and create a {{summary_type}} summary:\n\n"
        "Title: {{title}}\n"
        "Abstract: {{abstract}}\n"
        "Content: {{content}}\n\n"
        "{{custom_instructions}}"
    ),
    # Custom template variables
    template_variables={
        "analysis_focus": "practical applications",
        "target_audience": "researchers and practitioners"
    }
)

# Initialize flow
flow = SummaryFlow(config)

# Generate summary
result = flow.execute(knowledge_item)
print(result["summary"])
```

**Configuration Options:**
- `summary_type`: "brief", "comprehensive", "technical", "executive"
- `output_format`: "narrative", "structured", "bullet_points"
- `max_summary_length`: Maximum word count for summary
- `include_key_findings`: Include key findings section
- `include_methodology`: Include methodology overview
- `include_results`: Include main results
- `include_implications`: Include practical implications
- `focus_on_quantitative_aspects`: Emphasize quantitative content
- `highlight_innovations`: Highlight innovative contributions
- `include_limitations`: Include limitations and caveats

### 2. Q&A Flow (`simple_flow_examples.py`)

The `QAFlow` generates questions and answers from knowledge items for knowledge extraction with enhanced prompt engineering.

**Features:**
- Multiple difficulty levels
- Different question categories
- Customizable number of questions
- Enhanced template-based prompt system
- Custom template variables support
- Focus on insights and practical applications
- Generic KnowledgeItem support

**Enhanced Usage:**
```python
from quantmind.flow import QAFlow
from quantmind.config import QAFlowConfig

config = QAFlowConfig(
    llm_type="openai",
    api_key="your-api-key",
    num_questions=5,
    # Custom prompt template
    prompt_template=(
        "{{system_prompt}}\n\n"
        "Generate {{num_questions}} Q&A pairs for:\n"
        "Title: {{title}}\n"
        "Content Type: {{content_type}}\n"
        "Categories: {{categories}}\n"
        "{{custom_instructions}}"
    ),
    template_variables={"special_focus": "implementation challenges"}
)

flow = QAFlow(config)
qa_pairs = flow.execute(knowledge_item)
```

## Running Examples

### Prerequisites

1. Install QuantMind:
```bash
cd quant-mind
uv pip install -e .
```

2. Set up API key (for LLM flows):
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Running Content Summary Examples

```bash
cd quant-mind
python examples/flow/content_summary_example.py
```

This will run all content summary examples:
- Basic summary generation with KnowledgeItem
- Structured summary with custom templates
- Brief summary generation
- Executive summary generation
- Custom instructions and template variables example
- Backward compatibility with Paper objects

### Running Other Flow Examples

```bash
cd quant-mind
python examples/flow/simple_flow_examples.py
```

This will run all flow framework examples:
- Q&A Flow with enhanced prompt engineering
- Summary Flow with template variables
- Flow comparison and template debugging
- Backward compatibility demonstration

## Example Output

### Basic Summary Example
```
=== Basic Content Summary Example ===

Content: Deep Reinforcement Learning for Portfolio Optimization
Type: research_paper
Authors: John Smith, Jane Doe, Bob Johnson
Abstract: We propose a novel deep reinforcement learning approach for portfolio optimization...

==================================================

Generated Summary:
------------------------------
This research presents a novel deep reinforcement learning approach for portfolio optimization...

Word count: 245
Summary type: comprehensive
Content type: research_paper
```

### Enhanced Template Example
```
=== Flow Comparison & Template Demo ===

1. Q&A Flow - Generates questions and answers for knowledge extraction
   Template variables available: 12 (title, abstract, content, etc.)
   Generated 3 Q&A pairs

2. Summary Flow - Generates comprehensive content summaries
   Generated 245 word summary

Key Features Demonstrated:
- Enhanced prompt engineering with {{variable}} templates
- Generic KnowledgeItem support (not just Paper)
- Backward compatibility with existing Paper objects
- Custom template variables and instructions
- Template variable preview and debugging
```

## Configuration Examples

### Brief Summary for Quick Overview
```python
config = SummaryFlowConfig(
    summary_type="brief",
    max_summary_length=200,
    output_format="narrative"
)
```

### Technical Summary with Custom Template
```python
config = SummaryFlowConfig(
    summary_type="technical",
    max_summary_length=800,
    output_format="structured",
    focus_on_quantitative_aspects=True,
    include_methodology=True,
    include_limitations=True,
    # Custom template with variables
    prompt_template=(
        "{{system_prompt}}\n\n"
        "Technical analysis of {{content_type}}:\n"
        "Title: {{title}}\n"
        "Focus: {{analysis_focus}}\n"
        "Content: {{content}}\n\n"
        "{{custom_instructions}}"
    ),
    template_variables={
        "analysis_focus": "algorithmic trading applications"
    }
)
```

### Executive Summary for Business Users
```python
config = SummaryFlowConfig(
    summary_type="executive",
    max_summary_length=500,
    output_format="bullet_points",
    focus_on_quantitative_aspects=False,
    include_methodology=False,
    include_implications=True,
    template_variables={
        "target_audience": "executive decision makers",
        "business_focus": "competitive advantages"
    }
)
```

## Advanced Prompt Engineering

### Custom Instructions and Template Variables

```python
config = SummaryFlowConfig(
    custom_instructions="""
    Focus on practical trading implications.
    Highlight novel trading strategies.
    Consider implementation challenges.
    Emphasize competitive advantages.
    """,
    template_variables={
        "business_focus": "quantitative trading applications",
        "target_audience": "portfolio managers",
        "competitive_context": "hedge funds"
    }
)
```

### Custom Prompt Template

```python
config = QAFlowConfig(
    prompt_template=(
        "{{system_prompt}}\n\n"
        "You are analyzing {{content_type}} content.\n"
        "Generate {{num_questions}} questions about: {{title}}\n\n"
        "Focus Areas: {{question_categories}}\n"
        "Special Focus: {{special_focus}}\n"
        "Target Audience: {{target_audience}}\n\n"
        "Content:\n{{content}}\n\n"
        "{{custom_instructions}}\n\n"
        "Return structured JSON response."
    ),
    template_variables={
        "special_focus": "implementation challenges",
        "target_audience": "quantitative researchers"
    }
)
```

## Key Flow Features

### 1. Enhanced Prompt Engineering
- Template-based prompts with {{variable}} syntax
- Custom template variables
- Flexible system prompt configuration
- User-friendly template debugging

### 2. Generic Content Support
- Works with any KnowledgeItem (papers, articles, reports)
- Backward compatible with Paper objects
- Content-type aware processing

### 3. Debugging and Development
- Template variable preview: `flow.get_template_variables(item)`
- Prompt preview: `flow.get_prompt_preview(item)`
- Template validation: `flow.validate_template()`

## Testing

Run the flow tests:

```bash
cd quant-mind
python -m pytest tests/flow/ -v
```
