# Podcast Flow Examples

This directory contains examples demonstrating how to use the `PodcastFlow` class to generate podcast scripts from summary input, following the same structure as other custom flows in this project.

## Structure

```
03_podcast_flow/
├── README.md                           # This documentation file
├── pipeline.py                         # Main demo pipeline
├── config.yaml                         # Configuration file
└── flows/
    └── podcast_flow/
        ├── __init__.py                 # Package initialization
        ├── flow.py                     # Main flow implementation
        └── prompts.yaml                # Prompt templates
```

## Quick Start

```bash
# Run the demo pipeline
python pipeline.py

# Or import and use in your code
from flows.podcast_flow.flow import PodcastFlow
```

## Configuration

The flow is configured through `config.yaml`:

```yaml
flows:
  podcast_flow:
    type: "podcast"
    config:
      name: "podcast_flow"
      prompt_templates_path: "flows/podcast_flow/prompts.yaml"
      llm_blocks:
        intro_generator:
          model: "gpt-4o-mini"
          temperature: 0.7
          max_tokens: 300
        main_generator:
          model: "gpt-4o-mini"
          temperature: 0.5
          max_tokens: 1000
        outro_generator:
          model: "gpt-4o-mini"
          temperature: 0.7
          max_tokens: 300
```

## Usage

```python
from flows.podcast_flow.flow import PodcastFlow
from quantmind.config.settings import load_config

# Load configuration
settings = load_config("config.yaml")
config = settings.flows["podcast_flow"]

# Create and run flow
flow = PodcastFlow(config)
script = flow.run(
    summary="Your summary text here...",
    intro="AI in healthcare",
    outro="Future of technology"
)
```

## Features

- **Flexible Input**: Accepts summary text and optional intro/outro hints
- **LLM Integration**: Uses configured LLM blocks for content generation
- **Template System**: Supports customizable prompt templates via YAML
- **Structured Output**: Returns organized script sections (intro, main, outro)
- **Fallback Support**: Gracefully handles missing LLM blocks or templates

## Output Format

The flow returns a dictionary with:
```python
{
    "intro": "Generated intro content...",
    "main": "Generated main content...",
    "outro": "Generated outro content..."
}
```

## Prompt Templates

The flow uses three main prompt templates:

1. **intro_prompt**: Generates engaging podcast introductions
2. **main_prompt**: Converts summaries into conversational podcast content
3. **outro_prompt**: Creates effective podcast conclusions

## Environment Setup

Set your API keys as environment variables:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

## Examples

The `pipeline.py` file demonstrates:
- Loading configuration from YAML
- Initializing the flow
- Running with multiple sample inputs
- Error handling for missing API keys

## Key Takeaways

- **Consistent Structure**: Follows the same pattern as other custom flows
- **YAML Configuration**: Easy to modify without changing code
- **Template System**: Flexible prompt management
- **Error Handling**: Graceful fallbacks for missing resources
- **Extensible**: Easy to customize for different podcast styles
