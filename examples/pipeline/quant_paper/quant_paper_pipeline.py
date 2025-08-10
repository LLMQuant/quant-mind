from quantmind.config import Setting
from quantmind.parsers.llama_parser import LlamaParser
from quantmind.sources.arxiv_source import ArxivSource
from quantmind.storage.local_storage import LocalStorage


# TODO(whisper): Continuously optimize the interface design to achieve a more elegant user usage.
def quant_paper_pipeline():
    """Quant paper pipeline.

    Source --> Parser --> Storage --> SummaryFlow
    """
    setting = Setting.from_yaml("./qpa.yaml")

    local_storage = LocalStorage(setting.storage)
    arxiv_source = ArxivSource(setting.source)
    llama_parser = LlamaParser(setting.parser)

    # Get raw info from source.
    papers = arxiv_source.search(
        query="Large Language Models and Quantitative Finance", max_results=5
    )

    # Store raw info to local storage.
    local_storage.process_knowledges(papers)

    # Parse paper with LlamaParser.
    # TODO(whisper): Add a batch parse operation.
    for paper in papers:
        paper = llama_parser.parse_paper(paper)
        local_storage.store_knowledge(paper)


if __name__ == "__main__":
    quant_paper_pipeline()
