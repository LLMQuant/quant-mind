# Environment Setup Guide

QuantMind now uses modern environment management with dotenv support for handling API keys and configuration.

## Quick Start

### 1. Create Environment File

Create a `.env` file in your project root:

```bash
# Automatically create a sample .env file
python -c "from quantmind.utils.env import create_sample_env_file; create_sample_env_file()"
```

### 2. Configure API Keys

Edit the `.env` file with your actual API keys:

```bash
# QuantMind Configuration
LLAMA_CLOUD_API_KEY=your_actual_llama_cloud_api_key
OPENAI_API_KEY=your_actual_openai_api_key

# Optional: QuantMind Settings
QUANTMIND_LOG_LEVEL=INFO
QUANTMIND_DATA_DIR=./data
QUANTMIND_MAX_WORKERS=4
```

### 3. Use in Code

QuantMind automatically loads your environment configuration:

```python
from quantmind.utils.env import get_llama_cloud_api_key, load_environment

# Load environment (including .env files)
load_environment()

# Get API keys with modern approach
api_key = get_llama_cloud_api_key(required=False)

# Use in configuration
from quantmind.config.parsers import LlamaParserConfig
config = LlamaParserConfig(
    api_key=get_llama_cloud_api_key(required=False) or "demo_key",
    result_type="markdown",
    parsing_mode="fast"
)
```

## Environment File Locations

QuantMind automatically searches for `.env` files in these locations:

1. **Current directory**: `./env`
2. **Parent directory**: `../.env`
3. **User config**: `~/.quantmind/.env`

## Supported Environment Variables

### API Keys
- `LLAMA_CLOUD_API_KEY` - LlamaParse Cloud API key
- `OPENAI_API_KEY` - OpenAI API key

### QuantMind Settings
- `QUANTMIND_LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR)
- `QUANTMIND_DATA_DIR` - Data storage directory
- `QUANTMIND_TEMP_DIR` - Temporary files directory

### Workflow Settings
- `QUANTMIND_MAX_WORKERS` - Maximum worker threads
- `QUANTMIND_RETRY_ATTEMPTS` - Retry attempts for failed operations
- `QUANTMIND_TIMEOUT` - Operation timeout in seconds

### ArXiv Settings
- `QUANTMIND_ARXIV_MAX_RESULTS` - Maximum results from ArXiv searches

## Migration from Old Approach

If you were using direct environment variables or `os.getenv()`:

### Before (Old Approach)
```python
import os
api_key = os.getenv("LLAMA_CLOUD_API_KEY", "demo_key")
```

### After (Modern Approach)
```python
from quantmind.utils.env import get_llama_cloud_api_key, load_environment

# Load environment configuration
load_environment()

# Get API key with better error handling
api_key = get_llama_cloud_api_key(required=False) or "demo_key"
```

## Benefits

### ✅ **Modern Best Practices**
- Industry-standard dotenv approach
- Automatic environment discovery
- Better error handling and logging

### ✅ **Developer Experience**
- Clear setup instructions
- Sample configuration generation
- Automatic .env file loading

### ✅ **Security**
- API keys stored in .env files (not in code)
- .env files can be gitignored
- Environment variable precedence

### ✅ **Flexibility**
- Supports both .env files and system environment variables
- Multiple file location search
- Optional vs required API keys

## Best Practices

1. **Never commit .env files**: Add `.env` to your `.gitignore`
2. **Use .env.example**: Share template with placeholder values
3. **Environment precedence**: System env vars override .env files
4. **Error handling**: Use `required=False` for optional API keys
5. **Documentation**: Document all environment variables in your .env.example

## Troubleshooting

### .env file not loading?
- Check file location (current directory, parent, ~/.quantmind/)
- Verify file name is exactly `.env`
- Check file permissions

### API key not found?
- Verify the exact variable name (e.g., `LLAMA_CLOUD_API_KEY`)
- Check for typos in .env file
- Try using system environment variable as fallback

### Import errors?
- Ensure `python-dotenv` is installed: `pip install python-dotenv`
- Check that QuantMind is properly installed

## Examples

See `examples/parser/simple_llama_parser.py` for a complete example of the modern environment management approach.
