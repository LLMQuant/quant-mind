from pathlib import Path
import yaml
from autoscholar.crawler.github_crawler import GithubCrawler

def main():
    config_path = Path(__file__).parent / "config_sample.yaml"
    crawler = GithubCrawler.from_config_file(config_path)
    crawler.run()

if __name__ == "__main__":
    main()