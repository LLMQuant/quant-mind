from pathlib import Path
import yaml
from autoscholar.crawler.arxiv_crawler import ArxivCrawler


def main():
    """Run the arXiv crawler using configuration from config_sample.yaml."""
    config_path = Path(__file__).parent / "config_sample.yaml"
    crawler = ArxivCrawler.from_config_file(config_path)
    crawler.run()


if __name__ == "__main__":
    main()
