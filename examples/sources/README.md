# Sources Usage Examples

This directory contains comprehensive examples demonstrating how to use the QuantMind sources module, specifically the ArXiv source implementation.

## Files Overview

### 1. `basic_arxiv_usage.py`
Demonstrates fundamental ArXiv source operations:
- Basic paper search
- Retrieving papers by ArXiv ID
- Getting recent papers by timeframe
- Batch retrieval of multiple papers
- PDF download functionality
- Category-specific searches

**Run with:**
```bash
python examples/sources/basic_arxiv_usage.py
```

### 2. `advanced_configuration.py`
Shows advanced configuration options and scenarios:
- Finance-focused research configuration
- AI/ML research configuration
- Production-ready settings
- Loading configuration from YAML
- Configuration validation
- Comparing different configuration approaches

**Run with:**
```bash
python examples/sources/advanced_configuration.py
```

## Key Features Demonstrated

### Configuration Management
- **Pydantic-based validation**: All configurations use structured Pydantic models
- **Flexible initialization**: Accept both dict and config objects
- **Environment-specific settings**: Examples for development, research, and production
- **YAML support**: Load configurations from external files

### Content Filtering
- **Category filtering**: Include/exclude specific arXiv categories
- **Quality controls**: Minimum abstract length requirements
- **Content validation**: Ensure papers meet quality standards

### Download Management
- **Configurable downloads**: Enable/disable PDF downloads
- **Custom directories**: Specify download locations
- **Batch downloads**: Download multiple papers efficiently
- **Error handling**: Robust error handling for failed downloads

### Rate Limiting
- **Respectful usage**: Built-in rate limiting to respect arXiv's servers
- **Configurable rates**: Adjust request frequency based on use case
- **Timeout handling**: Configurable timeouts for reliability

## Configuration Examples

### Basic Configuration
```python
from quantmind.config.sources import ArxivSourceConfig
from quantmind.sources.arxiv_source import ArxivSource

# Simple configuration
config = ArxivSourceConfig(
    max_results=10,
    download_pdfs=True,
    download_dir="./papers"
)

source = ArxivSource(config=config)
```

### Finance Research Configuration
```python
config = ArxivSourceConfig(
    include_categories=["q-fin.ST", "q-fin.TR", "q-fin.PM"],
    min_abstract_length=150,
    requests_per_second=0.5,
    sort_by="submittedDate"
)
```

### Production Configuration
```python
config = ArxivSourceConfig(
    max_results=100,
    timeout=60,
    retry_attempts=3,
    requests_per_second=0.5,
    min_abstract_length=200,
    download_pdfs=True
)
```

## Usage Patterns

### 1. Basic Search
```python
source = ArxivSource()
papers = source.search("machine learning", max_results=5)
```

### 2. Timeframe Queries
```python
# Get papers from last 7 days in AI categories
papers = source.get_by_timeframe(
    days=7,
    categories=["cs.AI", "cs.LG"]
)
```

### 3. Batch Retrieval
```python
paper_ids = ["1706.03762", "1512.03385"]
papers = source.get_batch(paper_ids)
```

### 4. PDF Downloads
```python
config = ArxivSourceConfig(download_pdfs=True, download_dir="./pdfs")
source = ArxivSource(config=config)
papers = source.search("neural networks", max_results=3)
paths = source.download_papers_pdfs(papers)
```

## Best Practices

### 1. Rate Limiting
Always use appropriate rate limiting to be respectful to arXiv:
```python
config = ArxivSourceConfig(requests_per_second=1.0)  # 1 request per second
```

### 2. Content Quality
Filter for high-quality content:
```python
config = ArxivSourceConfig(
    min_abstract_length=100,  # Ensure substantial abstracts
    include_categories=["relevant", "categories"]  # Focus on relevant areas
)
```

### 3. Error Handling
Always handle potential errors:
```python
try:
    papers = source.search(query)
    if not papers:
        print("No papers found")
except Exception as e:
    logger.error(f"Search failed: {e}")
```

### 4. Configuration Validation
Validate configurations before use:
```python
source = ArxivSource(config=config)
if source.validate_config():
    # Proceed with operations
    pass
else:
    # Handle invalid configuration
    pass
```

## Advanced Features

### 1. Custom Filtering
Implement additional filtering logic:
```python
papers = source.search(query, max_results=100)
filtered_papers = [p for p in papers if custom_filter(p)]
```

### 3. Batch Processing
Process papers in batches for efficiency:
```python
def process_batch(papers):
    # Process batch of papers
    paths = source.download_papers_pdfs(papers)
    return paths

papers = source.search(query, max_results=50)
batch_size = 10
for i in range(0, len(papers), batch_size):
    batch = papers[i:i+batch_size]
    process_batch(batch)
```

## Notes

- **ArXiv Compliance**: All examples follow arXiv's API usage guidelines
- **Error Handling**: Comprehensive error handling for network issues
- **Logging**: Built-in logging for debugging and monitoring
- **Testing**: Examples include test scenarios for validation
- **Performance**: Optimized for both small queries and large-scale research

## Requirements

To run these examples, ensure you have:
- `arxiv` Python package for API access
- `requests` for PDF downloads
- `pydantic` for configuration validation
- `pyyaml` for YAML configuration support

Install requirements:
```bash
pip install arxiv requests pydantic pyyaml
```
