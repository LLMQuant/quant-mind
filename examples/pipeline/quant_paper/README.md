# Quant Paper Agent - Production Example

A production-level pipeline demonstrating intelligent quantitative finance paper processing using Google's Gemini 2.5 Pro & Flash models.

## Features

🔍 **Smart Paper Discovery**: Searches ArXiv for cutting-edge quantitative finance research
📄 **Advanced PDF Parsing**: Uses LlamaParser for high-quality content extraction
🧠 **Intelligent Summarization**: Gemini 2.5 Flash for chunks, Pro for synthesis
❓ **Thoughtful QA Generation**: Creates engaging questions for web display
💾 **Rich Metadata Storage**: Uses `meta_info` pattern for extensible data storage

## Architecture

This example demonstrates the **meta_info best practice** for storing Flow-generated content:

```python
# QA Flow stores results in meta_info
paper.meta_info["qa_data"] = {
    "questions": [{"id": "q_1", "question": "...", "category": "methodology"}],
    "generated_at": "2024-01-15T10:30:00",
    "model_used": "gemini-2.5-pro",
    "question_count": 7
}

# Summary Flow stores results in meta_info
paper.meta_info["summary"] = "Comprehensive summary text..."
```

## Setup

### 1. Environment Variables

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export LLAMA_CLOUD_API_KEY="your-llamaparse-key"
```

### 2. Install Dependencies

```bash
# From project root
uv pip install -e .
```

### 3. Run Pipeline

```bash
cd examples/pipeline/quant_paper
python pipeline.py
```

## Configuration

### Gemini Models Used

- **Gemini 2.5 Flash**: Cost-effective for chunk summarization and question refinement
- **Gemini 2.5 Pro**: High-quality for final summary synthesis and question generation

### Flow Configuration

```yaml
flows:
  summary_flow:
    llm_blocks:
      cheap_summarizer:
        provider: "google"
        model: "gemini-2.5-flash"
      powerful_combiner:
        provider: "google"
        model: "gemini-2.5-pro"

  qa_flow:
    max_questions: 7
    question_depth: "deep"
    focus_areas: ["methodology", "findings", "implications", "applications", "limitations"]
```

## Pipeline Workflow

1. **Paper Discovery**: Search ArXiv with finance-focused queries
2. **Content Extraction**: Download PDF and parse with LlamaParser
3. **Intelligent Summarization**: Two-stage process with Gemini models
4. **QA Generation**: Create thoughtful questions for engagement
5. **Metadata Storage**: Store all results in `paper.meta_info`
6. **Persistence**: Save enriched paper with full metadata

## Output Structure

### Stored Paper Data

```json
{
  "title": "Paper Title",
  "abstract": "Paper abstract...",
  "content": "Full extracted text...",
  "meta_info": {
    "summary": "Intelligent summary...",
    "summary_generated_at": "2024-01-15T10:30:00",
    "qa_data": {
      "questions": [
        {
          "id": "q_1",
          "question": "How does the proposed methodology handle market volatility?",
          "category": "methodology",
          "difficulty": "intermediate"
        }
      ],
      "generated_at": "2024-01-15T10:31:00",
      "model_used": "gemini-2.5-pro",
      "question_count": 7,
      "focus_areas": ["methodology", "findings", "implications"]
    },
    "storage_path": "/path/to/stored/paper.json"
  }
}
```

### Generated Questions

Questions are categorized for web display:

- **Category**: methodology, findings, implications, applications, limitations
- **Difficulty**: basic, intermediate, advanced
- **Structure**: Ready for web rendering

## Best Practices Demonstrated

### 1. Meta_info Pattern ✅

Store Flow outputs in `paper.meta_info` rather than extending Paper model:

```python
# ✅ Good: Flexible storage
paper.meta_info["qa_data"] = qa_results
paper.meta_info["custom_analysis"] = analysis_results

# ❌ Avoid: Rigid model extension
class Paper:
    qa_questions: List[Dict] = None  # Breaks separation of concerns
```

### 2. Production Configuration ✅

- Use environment variables for API keys
- Separate models for different tasks (Flash vs Pro)
- Reasonable token limits and temperatures
- Error handling and retries

### 3. Comprehensive Logging ✅

- Structured logging throughout pipeline
- Progress indicators for user experience
- Error context for debugging

## Accessing Stored Data

```python
from quantmind.models.paper import Paper

# Load stored paper
paper = Paper.load_from_file("data/knowledges/arxiv_id.json")

# Access generated content
summary = paper.meta_info.get("summary")
qa_data = paper.meta_info.get("qa_data", {})
questions = qa_data.get("questions", [])

# Display questions on web
for q in questions:
    print(f"Q: {q['question']}")
    print(f"Category: {q['category']} | Level: {q['difficulty']}")
```

## Extending the Pipeline

### Adding New Flows

1. Create flow in `flows/new_flow/`
2. Register configuration with `@register_flow_config`
3. Store results in `paper.meta_info["new_flow_data"]`
4. Update pipeline to run new flow

### Custom Processing

```python
# Add custom analysis
custom_flow = CustomAnalysisFlow(config)
analysis_result = custom_flow.run(paper)

# Store in meta_info
paper.meta_info["custom_analysis"] = analysis_result
paper.meta_info["custom_analysis_timestamp"] = datetime.now().isoformat()
```

## Performance Notes

- **Gemini 2.5 Flash**: ~2-3 seconds per chunk summary
- **Gemini 2.5 Pro**: ~5-8 seconds for final synthesis
- **Total Pipeline**: ~2-5 minutes per paper (depending on PDF size)
- **Cost Optimization**: Flash for repetitive tasks, Pro for complex reasoning

## Troubleshooting

### Common Issues

1. **Missing API Keys**: Set `GOOGLE_API_KEY` and `LLAMA_CLOUD_API_KEY`
2. **PDF Parse Failures**: Check LlamaParser quota and PDF quality
3. **Model Errors**: Verify Gemini API access and quota
4. **Storage Issues**: Ensure write permissions in `./data` directory

### Debug Mode

```python
import logging
logging.getLogger("quantmind").setLevel(logging.DEBUG)
```

---

This example showcases production-ready quantitative finance paper processing with modern AI models and extensible metadata storage patterns.
