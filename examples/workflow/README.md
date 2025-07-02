# QuantMind Workflow Examples

This directory contains examples demonstrating the various workflow components in QuantMind.

## Available Workflows

### 1. Paper Summary Workflow (`paper_summary_example.py`)

The `PaperSummaryWorkflow` generates comprehensive summaries of research papers using LLM technology.

**Features:**
- Multiple summary types: brief, comprehensive, technical, executive
- Flexible output formats: narrative, structured (JSON), bullet points
- Configurable content focus (quantitative aspects, innovations, limitations)
- Custom instructions support
- Convenience methods for common summary types

**Usage:**
```python
from quantmind.workflow.paper_summary_workflow import PaperSummaryWorkflow
from quantmind.config.workflows import PaperSummaryWorkflowConfig

# Create configuration
config = PaperSummaryWorkflowConfig(
    llm_type="openai",
    llm_name="gpt-4o",
    api_key="your-api-key",
    summary_type="comprehensive",
    max_summary_length=600,
    output_format="narrative"
)

# Initialize workflow
workflow = PaperSummaryWorkflow(config)

# Generate summary
result = workflow.execute(paper)
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

### 2. Q&A Workflow (`simple_workflow_examples.py`)

The `QAWorkflow` generates questions and answers from research papers for knowledge extraction.

**Features:**
- Multiple difficulty levels
- Different question categories
- Customizable number of questions
- Focus on insights and practical applications

### 3. Analyzer Workflow (`simple_workflow_examples.py`)

The `AnalyzerWorkflow` extracts structured tags and generates summaries from papers.

**Features:**
- Primary and secondary tag extraction
- Methodology and results summaries
- Confidence scoring
- Configurable tag categories

## Running Examples

### Prerequisites

1. Install QuantMind:
```bash
cd quant-mind
uv pip install -e .
```

2. Set up API key (for LLM workflows):
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Running Paper Summary Examples

```bash
cd quant-mind
python examples/workflow/paper_summary_example.py
```

This will run all paper summary examples:
- Basic summary generation
- Structured summary output
- Brief summary generation
- Executive summary generation
- Custom instructions example

### Running Other Workflow Examples

```bash
cd quant-mind
python examples/workflow/simple_workflow_examples.py
```

## Example Output

### Basic Summary Example
```
=== Basic Paper Summary Example ===

Paper: Deep Reinforcement Learning for Portfolio Optimization
Authors: John Smith, Jane Doe, Bob Johnson
Abstract: We propose a novel deep reinforcement learning approach for portfolio optimization...

==================================================

Generated Summary:
------------------------------
This research paper presents a novel deep reinforcement learning approach for portfolio optimization...

Word count: 245
Summary type: comprehensive
```

### Structured Summary Example
```
=== Structured Summary Example ===

Structured Summary Results:
------------------------------
Key Findings: The proposed DRL approach achieves superior risk-adjusted returns...
Methodology: Multi-agent reinforcement learning framework with attention mechanisms...
Results: Sharpe ratio of 1.85, outperforming traditional methods...
Implications: Potential for real-world portfolio management applications...
Limitations: Requires significant computational resources...

Paper ID: 2023.12345
Output format: structured
```

## Configuration Examples

### Brief Summary for Quick Overview
```python
config = PaperSummaryWorkflowConfig(
    summary_type="brief",
    max_summary_length=200,
    output_format="narrative"
)
```

### Technical Summary for Researchers
```python
config = PaperSummaryWorkflowConfig(
    summary_type="technical",
    max_summary_length=800,
    output_format="structured",
    focus_on_quantitative_aspects=True,
    include_methodology=True,
    include_limitations=True
)
```

### Executive Summary for Business Users
```python
config = PaperSummaryWorkflowConfig(
    summary_type="executive",
    max_summary_length=500,
    output_format="bullet_points",
    focus_on_quantitative_aspects=False,
    include_methodology=False,
    include_implications=True
)
```

## Custom Instructions

You can provide custom instructions to tailor the summary generation:

```python
config = PaperSummaryWorkflowConfig(
    custom_instructions="""
    Focus on practical trading implications.
    Highlight novel trading strategies.
    Consider implementation challenges.
    Emphasize competitive advantages.
    """
)
```

## Testing

Run the workflow tests:

```bash
cd quant-mind
python -m pytest tests/workflow/test_paper_summary_workflow.py -v
```
