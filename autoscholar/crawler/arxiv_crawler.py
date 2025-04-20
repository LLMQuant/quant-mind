import json
import arxiv
import datetime
import requests
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from autoscholar.crawler.base_crawler import BaseCrawler
from autoscholar.utils.logger import setup_logger

# ArXiv-specific constants
ARXIV_URL = "http://arxiv.org/"
BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"

# Set up logger
logger = setup_logger(__name__)


@dataclass
class ArxivCrawlerConfig:
    """Configuration class for ArxivCrawler.
    
    Attributes:
    ----------
    output_dir : str
        Directory to save the crawled data
    max_results : int
        Maximum number of papers to fetch per query
    download_pdf : bool
        Whether to download PDF files
    keywords : Dict[str, Any]
        Dictionary of search keywords and filters
    """
    output_dir: str = "data"
    max_results: int = 10
    download_pdf: bool = False
    keywords: Dict[str, Any] = None

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ArxivCrawlerConfig':
        """Create an ArxivCrawlerConfig instance from a dictionary.
        
        Parameters:
        ----------
        config_dict : Dict[str, Any]
            Dictionary containing configuration settings
            
        Returns:
        -------
        ArxivCrawlerConfig
            Configured instance
        """
        return cls(
            output_dir=config_dict.get("output_dir", "data"),
            max_results=config_dict.get("max_results", 10),
            download_pdf=config_dict.get("download_pdf", False),
            keywords=config_dict.get("keywords", {})
        )


class ArxivCrawler(BaseCrawler):
    """Crawler for fetching papers from arXiv.

    This crawler uses the arXiv API to fetch papers based on queries
    and saves the data in a structured format.
    """

    def __init__(self, **kwargs):
        """Initialize the ArXiv crawler.

        Parameters:
        ----------
        **kwargs : Any
            Optional parameters that can be used by the crawler.
        """
        super().__init__(**kwargs)
        self.config = ArxivCrawlerConfig.from_dict(kwargs)
        self.all_results = {}

    def get_authors(self, authors, partial_author: bool = False) -> str:
        """Retrieve a formatted string of authors.

        Parameters:
        ----------
        authors : list
            List of author names.
        partial_author : bool, optional
            If True, return only the first three authors.

        Returns:
        -------
        str
            String of author names.
        """
        if not partial_author:
            return ", ".join(str(author) for author in authors)
        else:
            return ", ".join(str(author) for author in authors[:3])

    def _get_pdf_folder(self, topic: str, date: datetime.date) -> Path:
        """Get the folder path for storing PDFs based on topic and date.

        Parameters:
        ----------
        topic : str
            Paper topic
        date : datetime.date
            Paper publication date

        Returns:
        -------
        Path
            Path to the PDF folder
        """
        # Create folder structure: output_dir/arxiv/topic/YYYY-MM
        folder_path = Path(self.config.output_dir) / "arxiv" / topic / date.strftime("%Y-%m")
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path

    def _download_pdf(self, result: arxiv.Result, paper_key: str, topic: str) -> None:
        """Download PDF for a paper.

        Parameters:
        ----------
        result : arxiv.Result
            Paper result from arXiv API
        paper_key : str
            Paper key (ID without version)
        topic : str
            Topic name for categorization
        """
        try:
            # Get the appropriate folder based on topic and date
            pdf_folder = self._get_pdf_folder(topic, result.published.date())
            pdf_path = pdf_folder / f"{paper_key}.pdf"
            
            pdf_response = requests.get(result.pdf_url)
            with open(pdf_path, "wb") as f:
                f.write(pdf_response.content)
            logger.info(f"Downloaded PDF for {paper_key} to {pdf_path}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading PDF for {paper_key}: {e}")

    def _get_code_url(self, paper_id: str) -> Optional[str]:
        """Get code repository URL for a paper.

        Parameters:
        ----------
        paper_id : str
            arXiv paper ID

        Returns:
        -------
        Optional[str]
            Code repository URL if found, None otherwise
        """
        try:
            response = requests.get(f"{BASE_URL}{paper_id}").json()
            if "official" in response and response["official"]:
                return response["official"]["url"]
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Error getting code URL for {paper_id}: {e}")
        return None

    def _process_paper(self, result: arxiv.Result, topic: str) -> Dict[str, Any]:
        """Process a single paper result.

        Parameters:
        ----------
        result : arxiv.Result
            Paper result from arXiv API
        topic : str
            Topic name for categorization

        Returns:
        -------
        Dict[str, Any]
            Processed paper data
        """
        paper_id = result.get_short_id()
        paper_key = paper_id.split('v')[0]  # Remove version number
        paper_url = f"{ARXIV_URL}abs/{paper_key}"

        # Get code repository URL if available
        repo_url = self._get_code_url(paper_id)

        # Download PDF if enabled
        if self.config.download_pdf:
            self._download_pdf(result, paper_key, topic)

        return {
            "topic": topic,
            "title": result.title,
            "authors": self.get_authors(result.authors),
            "first_author": self.get_authors(result.authors, partial_author=True),
            "abstract": result.summary.replace("\n", " "),
            "url": paper_url,
            "code_url": repo_url,
            "category": result.primary_category,
            "publish_time": str(result.published.date()),
            "update_time": str(result.updated.date()),
            "comments": result.comment.replace("\n", " ") if result.comment else ""
        }

    def _fetch_papers(self, topic: str, query: str, max_results: int) -> None:
        """Fetch papers for a specific topic.

        Parameters:
        ----------
        topic : str
            Topic name for categorization
        query : str
            Search query string
        max_results : int
            Maximum number of papers to fetch
        """
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in search.results():
            paper_key = result.get_short_id().split('v')[0]
            self.all_results[paper_key] = self._process_paper(result, topic)
            logger.info(f"Processed paper: {result.title}")

    def _save_results(self) -> None:
        """Save all crawled results to a single JSON file."""
        if not self.all_results:
            logger.warning("No results to save")
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.date.today().strftime("%Y-%m-%d")
        output_path = output_dir / f"arxiv_papers_{today}.json"
        
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
        logger.info(f"Saved {len(self.all_results)} papers to {output_path}")

    def run(self, **kwargs) -> None:
        """Execute the arXiv crawler workflow."""
        logger.info("Starting arXiv crawler")

        keywords = self.config.keywords or {}
        max_results = self.config.max_results

        logger.info("Fetching data begin")
        for topic, keyword_info in keywords.items():
            if isinstance(keyword_info, dict) and "filters" in keyword_info:
                query = " OR ".join(keyword_info["filters"])
                topic_max_results = keyword_info.get("max_results", max_results)
            else:
                query = topic
                topic_max_results = max_results

            logger.info(f"Processing topic: {topic}, query: {query}")
            self._fetch_papers(topic, query, topic_max_results)

        self._save_results()
        logger.info("Fetching data end")
