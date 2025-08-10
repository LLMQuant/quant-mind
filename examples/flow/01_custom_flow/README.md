# Custom Flow Example

This example demonstrates how to create a simple custom flow using the new QuantMind flow architecture.

## Structure

```bash
examples/flow/01_custom_flow/
├── flows/
│   └── greeting_flow/
│       ├── prompts.yaml    # Prompt templates
│       └── flow.py         # Flow implementation
├── config.yaml             # Flow configuration
├── pipeline.py             # Entry point
├── .env.example            # Example .env file
└── README.md              # This file
```

## Key Components

### 1. Flow Configuration (`GreetingFlowConfig`)

- Extends `BaseFlowConfig`
- Defines resource requirements (LLM blocks)
- Uses Pydantic models for simplicity
- Use `register_flow_config` decorator to register the flow config

### 2. Flow Implementation (`GreetingFlow`)

- Extends `BaseFlow`
- Implements custom `run()` method
- Direct access to LLM blocks: `self._llm_blocks["greeter"]`
- Template rendering: `self._render_prompt("template_name", **vars)`

### 3. Prompt Templates (`prompts.yaml`)

- Jinja2 templates with `{{ variable }}` syntax
- Separated from code for easy editing
- Loaded dynamically

## Running the Demo

```bash
cd examples/flow/01_custom_flow
# Prepare the api-key .env file
cp .env.example .env
python pipeline.py
```

## Benefits Demonstrated

1. **Simple Configuration**: No complex schemas, just resources
2. **Code-based Logic**: Python orchestration instead of config-driven
3. **Direct Access**: No unnecessary wrapper methods
4. **Template Separation**: YAML-based prompts
5. **Type Safety**: Pydantic models configuration

This showcases the new architecture's core principle: **provide resources, implement logic in code**.
