import os
import logging
import argparse
import datetime
import requests
import yaml
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional


# Configure logging
logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class BaseCrawler(ABC):
    """Base class for implementing crawlers for different sources.

    This abstract class defines the interface and common functionality
    for crawlers that collect data from various sources.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the crawler with configuration.

        Parameters:
        ----------
        config : Dict[str, Any]
            Dictionary containing configuration settings.
        """
        self.config = config
        self.data_collector = []
        self.output_dir = config.get("output_dir", "data")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @abstractmethod
    def fetch_data(
        self, topic: str, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """Fetch data from the source.

        Parameters:
        ----------
        topic : str
            Topic name for categorization.
        query : str
            Search query string.
        max_results : int, optional
            Maximum number of results to fetch.

        Returns:
        -------
        Dict[str, Any]
            Dictionary containing the fetched data.
        """
        pass

    @abstractmethod
    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the fetched data.

        Parameters:
        ----------
        data : Dict[str, Any]
            Raw data fetched from the source.

        Returns:
        -------
        Dict[str, Any]
            Processed data.
        """
        pass

    @abstractmethod
    def save_data(self, data: Dict[str, Any], output_path: str) -> None:
        """Save the processed data.

        Parameters:
        ----------
        data : Dict[str, Any]
            Processed data to save.
        output_path : str
            Path to save the data.
        """
        pass

    def run(self) -> None:
        """Execute the crawler workflow."""
        logging.info(f"Starting crawler: {self.__class__.__name__}")

        keywords = self.config.get("keywords", {})
        max_results = self.config.get("max_results", 10)

        logging.info("Fetching data begin")
        for topic, keyword_info in keywords.items():
            if isinstance(keyword_info, dict) and "filters" in keyword_info:
                query = " OR ".join(keyword_info["filters"])
            else:
                query = topic

            logging.info(f"Processing topic: {topic}, query: {query}")
            data = self.fetch_data(topic, query, max_results)
            processed_data = self.process_data(data)
            self.data_collector.append(processed_data)

        logging.info("Fetching data end")

        # Save collected data
        if self.data_collector:
            today = datetime.date.today().strftime("%Y-%m-%d")
            output_path = os.path.join(self.output_dir, f"{today}.json")
            for data in self.data_collector:
                self.save_data(data, output_path)
            logging.info(f"Data saved to {output_path}")


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Parameters:
    ----------
    config_file : str
        Path to the configuration file.

    Returns:
    -------
    Dict[str, Any]
        Dictionary containing configuration settings.
    """
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logging.info(f"Loaded config = {config}")
    return config


def main():
    """Main function to parse arguments and run the crawler."""
    parser = argparse.ArgumentParser(description="Generic data crawler")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--crawler_type",
        type=str,
        required=True,
        help="Type of crawler to use (e.g., arxiv, github, etc.)",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)

    # Import and initialize the specified crawler type
    try:
        if args.crawler_type == "arxiv":
            from arxiv_crawler import ArxivCrawler

            crawler = ArxivCrawler(config)
        elif args.crawler_type == "github":
            from github_crawler import GithubCrawler

            crawler = GithubCrawler(config)
        # TODO: Add more crawler types as needed
        else:
            raise ValueError(f"Unsupported crawler type: {args.crawler_type}")

        crawler.run()

    except ImportError as e:
        logging.error(f"Failed to import crawler module: {e}")
    except Exception as e:
        logging.error(f"Error running crawler: {e}")


if __name__ == "__main__":
    main()
