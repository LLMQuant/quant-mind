import json
import datetime
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from autoscholar.crawler.base_crawler import BaseCrawler
from autoscholar.utils.logger import setup_logger

# GitHub-specific constants
GITHUB_API_URL = "https://api.github.com/search/repositories"

# Set up logger
logger = setup_logger(__name__)


@dataclass
class GithubCrawlerConfig:
    """Configuration class for GithubCrawler.
    
    Attributes:
    ----------
    output_dir : str
        Directory to save the crawled data
    max_results : int
        Maximum number of repositories to fetch per query
    github_token : str
        GitHub API token for authentication
    keywords : Dict[str, Any]
        Dictionary of search keywords and filters
    """
    output_dir: str = "data"
    max_results: int = 10
    github_token: str = None
    keywords: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GithubCrawlerConfig':
        """Create a GithubCrawlerConfig instance from a dictionary.
        
        Parameters:
        ----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration settings
            
        Returns:
        -------
        GithubCrawlerConfig
            Configured instance
        """
        return cls(
            output_dir=config_dict.get("output_dir", "data"),
            max_results=config_dict.get("max_results", 10),
            github_token=config_dict.get("github_token"),
            keywords=config_dict.get("keywords", {})
        )


class GithubCrawler(BaseCrawler):
    """Crawler for fetching repositories from GitHub.

    This crawler uses the GitHub API to fetch repositories based on queries
    and saves the data in a structured format.
    """

    def __init__(self, **kwargs):
        """Initialize the GitHub crawler.

        Parameters:
        ----------
        **kwargs : Any
            Optional parameters that can be used by the crawler.
        """
        super().__init__(**kwargs)
        self.config = GithubCrawlerConfig.from_dict(kwargs)
        self.all_results = {}

        # GitHub API token (optional)
        self.headers = {}
        if self.config.github_token:
            self.headers["Authorization"] = f"token {self.config.github_token}"

    def run(self, **kwargs) -> None:
        """Execute the GitHub crawler workflow."""
        logger.info(f"Starting GitHub crawler")

        keywords = self.config.keywords or {}
        max_results = self.config.max_results

        logger.info("Fetching data begin")
        for topic, keyword_info in keywords.items():
            if isinstance(keyword_info, dict) and "filters" in keyword_info:
                query = " OR ".join(keyword_info["filters"])
            else:
                query = topic

            logger.info(f"Processing topic: {topic}, query: {query}")
            self._fetch_repos(topic, query, max_results)

        # Save all results to a single JSON file
        self._save_all_results()
        logger.info("Fetching data end")

    def _fetch_repos(self, topic: str, query: str, max_results: int) -> None:
        """Fetch repositories for a specific topic.

        Parameters:
        ----------
        topic : str
            Topic name for categorization
        query : str
            Search query string
        max_results : int
            Maximum number of repositories to fetch
        """

        # Set up the search parameters
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": max_results,
        }

        # Fetch repositories from GitHub API
        try:
            response = requests.get(
                GITHUB_API_URL, params=params, headers=self.headers
            )
            response.raise_for_status()
            results = response.json()

            if results["total_count"] == 0:
                logger.info(f"No repositories found for query: {query}")
                return

            # Process each repository
            for repo in results["items"]:
                repo_id = str(repo["id"])
                repo_name = repo["full_name"]
                repo_url = repo["html_url"]
                repo_description = repo["description"] if repo["description"] else "No description"
                repo_stars = repo["stargazers_count"]
                repo_forks = repo["forks_count"]
                repo_language = repo["language"] if repo["language"] else "Not specified"
                repo_created = repo["created_at"].split("T")[0]  # Format as YYYY-MM-DD
                repo_updated = repo["updated_at"].split("T")[0]

                logger.info(f"Repository: {repo_name}, Stars: {repo_stars}, Language: {repo_language}")

                # Store repository data
                repo_data = {
                    "topic": topic,
                    "name": repo_name,
                    "description": repo_description,
                    "url": repo_url,
                    "stars": repo_stars,
                    "forks": repo_forks,
                    "language": repo_language,
                    "created_at": repo_created,
                    "updated_at": repo_updated
                }

                self.all_results[repo_id] = repo_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching repositories: {e}")
            return

    def _save_all_results(self) -> None:
        """Save all crawled results to a single JSON file."""
        if not self.all_results:
            logger.warning("No results to save")
            return

        # Create output directory if it doesn't exist
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to a single JSON file with current date
        today = datetime.date.today().strftime("%Y-%m-%d")
        output_path = output_dir / f"github_repos_{today}.json"
        
        # Load existing data if any
        if output_path.exists():
            with open(output_path, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        # Update with new data
        existing_data.update(self.all_results)

        # Write back to file
        with open(output_path, "w") as f:
            json.dump(existing_data, f, indent=2)
        logger.info(f"Saved all repositories to {output_path}")
