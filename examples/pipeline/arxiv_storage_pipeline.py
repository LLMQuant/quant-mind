from quantmind.config import ArxivSourceConfig, LocalStorageConfig
from quantmind.sources import ArxivSource
from quantmind.storage import LocalStorage


def arxiv_storage_pipeline():
    """Example of how to use the LocalStorage to store papers from Arxiv."""
    arxiv_source_config = ArxivSourceConfig(max_results=3, timeout=5)
    local_storage_config = LocalStorageConfig(storage_dir="./data")

    arxiv_source = ArxivSource(config=arxiv_source_config)
    local_storage = LocalStorage(config=local_storage_config)

    papers = arxiv_source.search("LLM in Quant Trading")
    local_storage.process_knowledges(papers)

    # You can also use the basic operation to deliver the same result
    # for paper in papers:
    #     pdf_url = paper.pdf_url
    #     if pdf_url:
    #         import requests
    #         response = requests.get(pdf_url)
    #         pdf_path = local_storage.store_raw_file(
    #             file_id=paper.get_primary_id(),
    #             content=response.content,
    #             file_extension=".pdf",
    #         )

    # Check if the pdf is stored
    for paper in papers:
        pdf_path = local_storage.get_raw_file(paper.get_primary_id())
        if pdf_path:
            print(f"PDF found at {pdf_path}")
        else:
            print(f"PDF not found for {paper.title}")


if __name__ == "__main__":
    arxiv_storage_pipeline()
