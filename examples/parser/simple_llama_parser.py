"""Simple LlamaParser usage example.

This example shows basic usage of the LlamaParser for parsing
individual PDF files or URLs using the new Pydantic configuration system
with modern dotenv-based environment management.
"""

import logging
from pathlib import Path

from colorama import Fore, Style

from quantmind.config.parsers import LlamaParserConfig, ParsingMode, ResultType
from quantmind.models.paper import Paper
from quantmind.parsers.llama_parser import LlamaParser
from quantmind.utils.env import get_llama_cloud_api_key, load_environment

# Set logging level to DEBUG
logging.basicConfig(level=logging.DEBUG)


def demo_file_parsing():
    """Demonstrate parsing a local PDF file."""
    print("=== File Parsing Demo ===\n")

    # Create parser with Pydantic configuration using modern env management
    config = LlamaParserConfig(
        api_key=get_llama_cloud_api_key(required=False) or "demo_key",
        result_type=ResultType.MD,
        parsing_mode=ParsingMode.FAST,
        max_file_size_mb=25,
    )
    parser = LlamaParser(config)

    # Example PDF path (would need to exist for real usage)
    pdf_path = "./examples/parser/test-pdf.pdf"

    print(f"Parser configuration:")
    print(
        f"- Result type: {parser.llama_config.result_type if isinstance(parser.llama_config.result_type, str) else parser.llama_config.result_type.value}"
    )
    print(
        f"- Parsing mode: {parser.llama_config.parsing_mode if isinstance(parser.llama_config.parsing_mode, str) else parser.llama_config.parsing_mode.value}"
    )
    print(f"- Max file size: {parser.llama_config.max_file_size_mb}MB")
    print()

    if Path(pdf_path).exists():
        print(f"Parsing file: {pdf_path}")
        try:
            # Parse the file
            content = parser.extract_from_file(pdf_path)
            print(
                Fore.GREEN
                + f"Successfully parsed {len(content)} characters"
                + Style.RESET_ALL
            )
            print(
                Fore.GREEN
                + f"Content preview: {content[:200]}..."
                + Style.RESET_ALL
            )

        except Exception as e:
            print(Fore.RED + f"Parsing error: {e}" + Style.RESET_ALL)
    else:
        print(Fore.RED + f"Demo PDF not found at: {pdf_path}" + Style.RESET_ALL)
        print(
            Fore.YELLOW
            + "In real usage, provide path to an existing PDF file."
            + Style.RESET_ALL
        )

    print()


def demo_url_parsing():
    """Demonstrate parsing a PDF from URL."""
    print("=== URL Parsing Demo ===\n")

    config = LlamaParserConfig(
        api_key=get_llama_cloud_api_key(required=False) or "demo_key",
        result_type=ResultType.TXT,
        parsing_mode=ParsingMode.BALANCED,
    )
    parser = LlamaParser(config)

    # Example PDF URL (ArXiv paper)
    pdf_url = "https://arxiv.org/pdf/2301.00001.pdf"

    print(f"Parsing URL: {pdf_url}")
    print("Note: This requires valid LLAMA_CLOUD_API_KEY")

    try:
        # Parse from URL
        content = parser.extract_from_url(pdf_url)
        print(
            Fore.GREEN
            + f"Successfully parsed {len(content)} characters"
            + Style.RESET_ALL
        )
        print(
            Fore.GREEN
            + f"Content preview: {content[:200]}..."
            + Style.RESET_ALL
        )

    except Exception as e:
        print(
            Fore.RED + f"Expected error without API key: {e}" + Style.RESET_ALL
        )

    print()


def demo_paper_parsing():
    """Demonstrate parsing a Paper object."""
    print("=== Paper Object Parsing Demo ===\n")

    # Create a Paper object
    paper = Paper(
        paper_id="2301.00001",
        title="Example Financial Research Paper",
        authors=["Author One", "Author Two"],
        abstract="This paper demonstrates quantitative methods...",
        pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
        categories=["q-fin.ST"],
    )

    # Create parser with advanced configuration
    config = LlamaParserConfig(
        api_key=get_llama_cloud_api_key(required=False) or "demo_key",
        result_type=ResultType.MD,
        parsing_mode=ParsingMode.PREMIUM,
        system_prompt=(
            "Extract key findings, methodology, and quantitative "
            "results from this financial research paper."
        ),
        system_prompt_append=(
            "Focus on statistical methods and financial metrics."
        ),
    )
    parser = LlamaParser(config)

    print(f"Paper: {paper.title}")
    print(f"Authors: {', '.join(paper.authors)}")
    print(f"PDF URL: {paper.pdf_url}")
    print()

    print("Parser configuration:")
    info = parser.get_parser_info()
    for key, value in info.items():
        if key != "config":  # Skip detailed config
            print(f"- {key}: {value}")
    print()

    try:
        # Parse the paper
        enriched_paper = parser.parse_paper(paper)

        if enriched_paper.content:
            print(Fore.GREEN + f"Parsing successful!" + Style.RESET_ALL)
            print(
                Fore.GREEN
                + f"Content length: {len(enriched_paper.content)} characters"
                + Style.RESET_ALL
            )
            print(
                Fore.GREEN
                + f"Parser info: {enriched_paper.meta_info.get('parser_info', {})}"
                + Style.RESET_ALL
            )
        else:
            print(
                Fore.RED
                + "No content extracted (expected without API key)"
                + Style.RESET_ALL
            )

    except Exception as e:
        print(
            Fore.RED + f"Expected error without API key: {e}" + Style.RESET_ALL
        )

    print()


def demo_configuration_options():
    """Demonstrate different configuration options."""
    print("=== Configuration Options Demo ===\n")

    # Get API key once for all configurations
    api_key = get_llama_cloud_api_key(required=False) or "demo_key"

    configurations = [
        {
            "name": "Fast Text Extraction",
            "config": LlamaParserConfig(
                api_key=api_key,
                result_type=ResultType.TXT,
                parsing_mode=ParsingMode.FAST,
                max_file_size_mb=50,
            ),
        },
        {
            "name": "Balanced Markdown",
            "config": LlamaParserConfig(
                api_key=api_key,
                result_type=ResultType.MD,
                parsing_mode=ParsingMode.BALANCED,
                max_file_size_mb=25,
            ),
        },
        {
            "name": "Premium with Custom Prompts",
            "config": LlamaParserConfig(
                api_key=api_key,
                result_type=ResultType.MD,
                parsing_mode=ParsingMode.PREMIUM,
                max_file_size_mb=15,
                system_prompt="Extract financial data and analysis",
                system_prompt_append="Include all numerical results",
            ),
        },
    ]

    for config_info in configurations:
        print(f"Configuration: {config_info['name']}")
        try:
            parser = LlamaParser(config_info["config"])
            info = parser.get_parser_info()
            print(f"- Result type: {info['result_type']}")
            print(f"- Parsing mode: {info['parsing_mode']}")
            print(f"- Max file size: {info['max_file_size_mb']}MB")
            if info.get("system_prompt"):
                print(f"- Custom prompt: Yes")
            print()

        except Exception as e:
            print(Fore.RED + f"Configuration error: {e}" + Style.RESET_ALL)
            print()


def main():
    """Run all demonstration examples."""
    print("QuantMind LlamaParser: Simple Usage Examples")
    print("=" * 45)
    print()

    # Load environment configuration (including .env files)
    env_loaded = load_environment()
    if env_loaded:
        print(
            Fore.GREEN
            + "🔧 Environment configuration loaded from .env file"
            + Style.RESET_ALL
        )

    # Check for API key using modern approach
    try:
        api_key = get_llama_cloud_api_key(required=False)
        if api_key and api_key != "demo_key":
            print("✅ LLAMA_CLOUD_API_KEY found - examples will use real API")
        else:
            print("⚠️  LLAMA_CLOUD_API_KEY not set - running in demo mode")
            print(
                "   💡 Tip: Create a .env file with LLAMA_CLOUD_API_KEY=your_key"
            )
            print("   Or set as environment variable for full functionality")
    except Exception as e:
        print(f"⚠️  Error loading API key: {e}")
        print("   Running in demo mode")
    print()

    # Run demonstrations
    demo_configuration_options()
    demo_file_parsing()
    demo_url_parsing()
    demo_paper_parsing()

    print("=" * 45)
    print(Fore.GREEN + "Examples completed!" + Style.RESET_ALL)
    print("\nNext steps:")
    print(
        Fore.YELLOW + "1. 🔑 Set up API key for real parsing:" + Style.RESET_ALL
    )
    print(
        Fore.YELLOW
        + "   • Create .env file: LLAMA_CLOUD_API_KEY=your_actual_key"
        + Style.RESET_ALL
    )
    print(
        Fore.YELLOW
        + "   • Or set environment variable: export LLAMA_CLOUD_API_KEY=your_key"
        + Style.RESET_ALL
    )
    print(Fore.YELLOW + "2. 📄 Try with your own PDF files" + Style.RESET_ALL)
    print(
        Fore.YELLOW
        + "3. ⚙️  Experiment with different parsing modes"
        + Style.RESET_ALL
    )
    print(
        Fore.YELLOW
        + "4. 🔄 See arxiv_llama_pipeline.py for full workflow integration"
        + Style.RESET_ALL
    )
    print(
        Fore.YELLOW
        + "\n💡 Run `python -c 'from quantmind.utils.env import create_sample_env_file; create_sample_env_file()'` to create a sample .env file"
        + Style.RESET_ALL
    )


if __name__ == "__main__":
    main()
