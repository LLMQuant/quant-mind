"""Simple demo: ArXiv source + LlamaParser integration.

Direct usage example:
1. Get "Attention Is All You Need" paper from ArXiv
2. Parse with LlamaParser (fast mode)
3. Show parsed content

No complex pipeline - just basic integration demo.
"""

from colorama import Fore, Style, init

from quantmind.config.parsers import LlamaParserConfig, ParsingMode, ResultType
from quantmind.parsers.llama_parser import LlamaParser
from quantmind.sources.arxiv_source import ArxivSource

# Initialize colorama
init(autoreset=True)


def main():
    """Direct ArXiv + LlamaParser demo."""
    print(
        f"{Fore.CYAN}{Style.BRIGHT}=== ArXiv + LlamaParser Simple Demo ==={Style.RESET_ALL}\n"
    )

    try:
        # 1. Get paper from ArXiv
        print(
            f"{Fore.YELLOW}📚 Step 1: Searching ArXiv for 'Attention Is All You Need'..."
        )
        arxiv_source = ArxivSource(
            config={
                "max_results": 1,
                "download_pdfs": True,
            }
        )

        papers = arxiv_source.search(
            query='ti:"Attention Is All You Need"', max_results=1
        )

        if not papers:
            print(f"{Fore.RED}❌ Paper not found")
            return

        paper = papers[0]
        print(f"{Fore.GREEN}✅ Found: {Style.BRIGHT}{paper.title}")
        print(f"{Fore.BLUE}   👥 Authors: {', '.join(paper.authors[:3])}")
        print(f"{Fore.BLUE}   🔗 PDF URL: {paper.pdf_url}")
        print()

        # 2. Parse with LlamaParser (fast, cheap mode)
        print(
            f"{Fore.YELLOW}🔧 Step 2: Parsing with LlamaParser (fast mode)..."
        )
        llama_config = LlamaParserConfig(
            result_type=ResultType.TXT,  # Cheapest
            parsing_mode=ParsingMode.FAST,  # Fastest
            max_file_size_mb=25,
        )

        llama_parser = LlamaParser(llama_config)
        parsed_paper = llama_parser.parse_paper(paper)

        # 3. Show results
        print(f"{Fore.GREEN}✅ Parsing completed!")
        print(
            f"{Fore.MAGENTA}   📊 Content length: {Style.BRIGHT}{len(parsed_paper.content):,} characters"
        )
        print(
            f"{Fore.MAGENTA}   ⚙️  Parser info: {parsed_paper.meta_info.get('parser_info', {})}"
        )
        print()
        print(f"{Fore.CYAN}📖 First 500 characters of parsed content:")
        print(f"{Fore.CYAN}{'-' * 50}")
        print(f"{Style.DIM}{parsed_paper.content[:500]}...")

    except Exception as e:
        print(f"{Fore.RED}❌ Error: {e}")
        print(
            f"{Fore.YELLOW}💡 Note: Requires LLAMA_CLOUD_API_KEY environment variable"
        )


if __name__ == "__main__":
    main()
