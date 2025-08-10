# Summary Flow Example

This example demonstrates the built-in SummaryFlow with custom chunking strategy and automatic API key management.

## Structure

```bash
examples/flow/02_summary_flow/
├── flows/
│   └── summary_flow/
│       ├── prompts.yaml    # Prompt templates
│       └── flow.py         # Mock LLM implementation (for testing)
├── config.yaml             # Flow configuration
├── mock_data.py            # Sample financial papers
├── pipeline.py             # Entry point
├── .env.example            # Example .env file
└── README.md              # This file
```

## Key Components

### 1. Built-in SummaryFlow

- Two-stage summarization: cheap model for chunks → powerful model for combination
- Flexible chunking strategies: size-based, custom, or disabled
- Cost-optimized mixture mode

### 2. Custom Chunking Strategy

- Demonstrates `ChunkingStrategy.BY_CUSTOM`
- User-defined chunking function (paragraph-based in demo)
- Runtime configuration update

### 3. Automatic API Key Management

- Environment variable resolution from `.env` file
- Smart provider inference (OpenAI models use `OPENAI_API_KEY`)
- Secure configuration without hardcoded keys

## Running the Demo

```bash
cd examples/flow/02_summary_flow
# Prepare the api-key .env file
cp ../../../.env.example .env
# Edit .env with your actual API keys
python pipeline.py
```

## Features Demonstrated

1. **Custom Chunking**: Paragraph-based instead of size-based splitting
2. **Two-stage Processing**: Cost-effective bulk processing + high-quality synthesis
3. **API Key Resolution**: Automatic environment variable discovery
4. **Template Separation**: YAML-based prompts for easy editing
5. **Type-safe Configuration**: Proper `SummaryFlowConfig` loading

## Configuration

```yaml
# config.yaml
flows:
  summary_demo:
    type: "summary"  # Uses built-in SummaryFlow
    config:
      name: "summary_flow"
      prompt_templates_path: "flows/summary_flow/prompts.yaml"
      use_chunking: true
      chunk_size: 1000
```

## Benefits Demonstrated

1. **Smart Resource Usage**: Cheap model for bulk work, powerful model for synthesis
2. **Flexible Chunking**: Easy to implement custom splitting strategies
3. **Secure Configuration**: Environment variable-based API key management
4. **Simple Integration**: Built-in flow with minimal configuration
5. **Template Flexibility**: External YAML prompt templates

This showcases the framework's principle: **powerful built-in flows with flexible customization options**.
