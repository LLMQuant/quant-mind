import os
import json
import datetime
import requests
from typing import Dict, List, Any, Optional

from base_crawler import BaseCrawler
from utils.logger import setup_logger

# GitHub-specific constants
GITHUB_API_URL = "https://api.github.com/search/repositories"
GITHUB_URL = "https://github.com/"

# Set up logger
logger = setup_logger(__name__)


class GithubCrawler(BaseCrawler):
    """Crawler for fetching repositories from GitHub.

    This crawler uses the GitHub API to fetch repositories based on queries
    and saves the data in a structured format.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the GitHub crawler.

        Parameters:
        ----------
        config : Dict[str, Any]
            Dictionary containing configuration settings.
        """
        super().__init__(config)
        self.GITHUB_API_URL = GITHUB_API_URL
        self.GITHUB_URL = GITHUB_URL

        # GitHub API token (optional)
        self.api_token = config.get("github_token", None)
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"token {self.api_token}"

    def fetch_data(
        self, topic: str, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """Fetch repositories from GitHub based on the query.

        Parameters:
        ----------
        topic : str
            Topic name for categorization.
        query : str
            Search query string.
        max_results : int, optional
            Maximum number of repositories to fetch.

        Returns:
        -------
        Dict[str, Any]
            Dictionary containing the fetched repositories.
        """
        content = {}

        today_month = datetime.date.today().strftime("%Y-%m")
        folder_path = os.path.join(self.output_dir, today_month)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
                self.GITHUB_API_URL, params=params, headers=self.headers
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            results = response.json()

            if results["total_count"] == 0:
                logger.info(f"No repositories found for query: {query}")
                return {topic: {}}

            # Process each repository
            for repo in results["items"]:
                repo_id = str(repo["id"])
                repo_name = repo["full_name"]
                repo_url = repo["html_url"]
                repo_description = (
                    repo["description"]
                    if repo["description"]
                    else "No description"
                )
                repo_stars = repo["stargazers_count"]
                repo_forks = repo["forks_count"]
                repo_language = (
                    repo["language"] if repo["language"] else "Not specified"
                )
                repo_created = repo["created_at"].split("T")[
                    0
                ]  # Format as YYYY-MM-DD
                repo_updated = repo["updated_at"].split("T")[0]

                logger.info(
                    f"Repository: {repo_name}, Stars: {repo_stars}, Language: {repo_language}"
                )

                # Format the repository data
                content[repo_id] = (
                    "|**{}**|**{}**|{}|[{}]({})|{}|{}|{}|\n".format(
                        repo_updated,
                        repo_name,
                        repo_description,
                        repo_language,
                        repo_url,
                        repo_stars,
                        repo_forks,
                        repo_created,
                    )
                )

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching repositories: {e}")
            return {topic: {}}

        return {topic: content}

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the fetched repository data.

        For GitHub repositories, we sort them by stars in descending order.

        Parameters:
        ----------
        data : Dict[str, Any]
            Raw data fetched from GitHub.

        Returns:
        -------
        Dict[str, Any]
            Processed data.
        """
        processed_data = {}

        # For each topic, sort repositories by stars
        for topic, repos in data.items():
            sorted_repos = {}
            for repo_id, repo_data in repos.items():
                sorted_repos[repo_id] = repo_data
            processed_data[topic] = sorted_repos

        return processed_data

    def save_data(self, data: Dict[str, Any], output_path: str) -> None:
        """Save the repository data to a JSON file.

        Parameters:
        ----------
        data : Dict[str, Any]
            Processed repository data to save.
        output_path : str
            Path to save the data.
        """
        # Check if file exists
        if not os.path.exists(output_path):
            with open(output_path, "w") as f:
                f.write("{}")

        # Load existing data
        with open(output_path, "r") as f:
            content = f.read()
            if not content:
                existing_data = {}
            else:
                existing_data = json.loads(content)

        # Update with new data
        for topic, repos in data.items():
            if topic in existing_data:
                existing_data[topic].update(repos)
            else:
                existing_data[topic] = repos

        # Write back to file
        with open(output_path, "w") as f:
            json.dump(existing_data, f)

        logger.info(f"Saved repository data to {output_path}")
