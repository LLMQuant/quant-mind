"""ArXiv source implementation for paper acquisition."""

import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import arxiv
import requests

from quantmind.config.sources import ArxivSourceConfig
from quantmind.models.paper import Paper
from quantmind.sources.base import BaseSource
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivSource(BaseSource[Paper]):
    """ArXiv source for academic paper acquisition.

    Provides access to arXiv papers through the arXiv API with support
    for searching, filtering, and retrieving paper metadata with optional PDF downloads.
    """

    def __init__(self, config: Optional[Union[ArxivSourceConfig, dict]] = None):
        """Initialize ArXiv source.

        Args:
            config: ArxivSourceConfig instance or dictionary with settings
        """
        # Convert dict to config if necessary
        if isinstance(config, dict):
            config = ArxivSourceConfig(**config)
        elif config is None:
            config = ArxivSourceConfig()

        super().__init__(config.model_dump())
        self.config = config
        self.client = arxiv.Client()
        # As the last request time is not set, the first request will not be rate limited
        self._last_request_time = 0.0

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search arXiv for papers matching the query.

        Args:
            query: Search query string (supports arXiv query syntax)
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects
        """
        try:
            self._rate_limit()

            search = arxiv.Search(
                query=query,
                max_results=min(max_results, self.config.max_results),
                sort_by=self.config.get_arxiv_sort_criterion(),
                sort_order=self.config.get_arxiv_sort_order(),
            )

            papers = []
            for result in self.client.results(search):
                paper = self._convert_arxiv_result(result)
                if self._should_include_paper(paper):
                    papers.append(paper)

            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers

        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []

    def get_by_id(self, paper_id: str) -> Optional[Paper]:
        """Get a specific paper by arXiv ID.

        Args:
            paper_id: ArXiv ID (e.g., '2301.12345' or 'cs.AI/1234567')

        Returns:
            Paper object if found, None otherwise
        """
        try:
            # Clean the arXiv ID
            clean_id = self._clean_arxiv_id(paper_id)

            search = arxiv.Search(id_list=[clean_id])
            results = list(self.client.results(search))

            if results:
                return self._convert_arxiv_result(results[0])
            return None

        except Exception as e:
            logger.error(f"Error fetching paper {paper_id}: {e}")
            return None

    def get_by_timeframe(
        self, days: int = 7, categories: Optional[List[str]] = None, **kwargs
    ) -> List[Paper]:
        """Get papers from arXiv within a specific timeframe.

        Args:
            days: Number of days to look back
            categories: Optional list of arXiv categories to filter by
            **kwargs: Additional filters (unused for arXiv)

        Returns:
            List of Paper objects from the timeframe
        """
        # Build query for recent papers
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        date_query = f"submittedDate:[{start_date.strftime('%Y%m%d')}* TO {end_date.strftime('%Y%m%d')}*]"

        if categories:
            # Build category query
            cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
            query = f"({cat_query}) AND {date_query}"
        else:
            query = date_query

        return self.search(query, max_results=self.config.max_results)

    def search_by_category(
        self, category: str, max_results: int = 50
    ) -> List[Paper]:
        """Search papers by arXiv category.

        Args:
            category: ArXiv category (e.g., 'cs.AI', 'stat.ML', 'q-fin.ST')
            max_results: Maximum number of results

        Returns:
            List of Paper objects
        """
        query = f"cat:{category}"
        return self.search(query, max_results)

    def search_financial_topics(self, max_results: int = 100) -> List[Paper]:
        """Search for papers related to quantitative finance.

        Args:
            max_results: Maximum number of results

        Returns:
            List of Paper objects from finance-related categories
        """
        # Finance-related arXiv categories and keywords
        finance_categories = [
            "q-fin.ST",
            "q-fin.TR",
            "q-fin.PM",
            "q-fin.RM",
            "q-fin.CP",
        ]
        finance_keywords = [
            "quantitative finance",
            "portfolio optimization",
            "risk management",
            "algorithmic trading",
            "financial markets",
            "machine learning finance",
        ]

        # Build comprehensive query
        cat_query = " OR ".join([f"cat:{cat}" for cat in finance_categories])
        keyword_query = " OR ".join(
            [f'all:"{keyword}"' for keyword in finance_keywords]
        )

        query = f"({cat_query}) OR ({keyword_query})"
        return self.search(query, max_results)

    def _convert_arxiv_result(self, result: arxiv.Result) -> Paper:
        """Convert arXiv result to Paper object.

        Args:
            result: ArXiv API result object

        Returns:
            Paper object
        """
        # Extract arXiv ID from URL
        arxiv_id = result.entry_id.split("/")[-1]

        # Convert categories to strings
        categories = [str(cat) for cat in result.categories]

        # Build paper object
        paper = Paper(
            arxiv_id=arxiv_id,
            title=result.title,
            abstract=result.summary,
            authors=[str(author) for author in result.authors],
            published_date=result.published,
            categories=categories,
            url=result.entry_id,
            pdf_url=result.pdf_url,
            source="arxiv",
            extraction_method="api",
            meta_info={
                "doi": result.doi,
                "journal_ref": result.journal_ref,
                "primary_category": result.primary_category,
                "comment": result.comment,
                "links": [link.href for link in result.links]
                if result.links
                else [],
            },
        )

        return paper

    def _clean_arxiv_id(self, arxiv_id: str) -> str:
        """Clean and normalize arXiv ID.

        Args:
            arxiv_id: Raw arXiv ID

        Returns:
            Cleaned arXiv ID
        """
        # Remove common prefixes and clean
        arxiv_id = arxiv_id.replace("arXiv:", "").replace("arxiv:", "")
        arxiv_id = arxiv_id.replace("http://arxiv.org/abs/", "")
        arxiv_id = arxiv_id.replace("https://arxiv.org/abs/", "")

        return arxiv_id.strip()

    def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        min_interval = 1.0 / self.config.requests_per_second

        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.info(
                f"Rate limiting: {time_since_last} seconds since last request, sleeping for {sleep_time} seconds"
            )
            time.sleep(sleep_time)

        self._last_request_time = time.time()

    def _should_include_paper(self, paper: Paper) -> bool:
        """Check if paper should be included based on config filters."""
        # Check abstract length
        if (
            paper.abstract
            and len(paper.abstract) < self.config.min_abstract_length
        ):
            return False

        # Check category filters
        if self.config.include_categories:
            if not any(
                cat in paper.categories
                for cat in self.config.include_categories
            ):
                return False

        if self.config.exclude_categories:
            if any(
                cat in paper.categories
                for cat in self.config.exclude_categories
            ):
                return False

        return True

    def download_pdf(
        self, paper: Paper, download_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Download PDF for a paper.

        Args:
            paper: Paper object with pdf_url
            download_dir: Directory to save PDF (uses config if not provided)

        Returns:
            Path to downloaded PDF file, None if failed
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for paper {paper.get_primary_id()}")
            return None

        # Determine download directory
        if download_dir is None:
            download_dir = self.config.download_dir

        if download_dir is None:
            logger.error("No download directory specified")
            return None

        download_dir = Path(download_dir)
        download_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        safe_title = re.sub(r"[^\w\s-]", "", paper.title)[:50]
        safe_title = re.sub(r"[-\s]+", "-", safe_title).strip("-")
        filename = f"{paper.get_primary_id()}_{safe_title}.pdf"
        file_path = download_dir / filename

        try:
            self._rate_limit()
            response = requests.get(
                paper.pdf_url,
                timeout=self.config.timeout,
                proxies=self.config.proxies,
            )
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"Downloaded PDF: {file_path}")
            return file_path

        except Exception as e:
            logger.error(
                f"Failed to download PDF for {paper.get_primary_id()}: {e}"
            )
            return None

    def download_papers_pdfs(self, papers: List[Paper]) -> List[Optional[Path]]:
        """Download PDFs for multiple papers.

        Args:
            papers: List of Paper objects

        Returns:
            List of file paths (None for failed downloads)
        """
        if not self.config.download_pdfs:
            logger.info("PDF download is disabled in config")
            return [None] * len(papers)

        paths = []
        for paper in papers:
            path = self.download_pdf(paper)
            paths.append(path)

        return paths

    def validate_config(self) -> bool:
        """Validate ArXiv source configuration.

        Returns:
            True if configuration is valid
        """
        try:
            # Test connection with a simple query
            test_search = arxiv.Search(query="test", max_results=1)
            list(self.client.results(test_search))
            return True
        except Exception as e:
            logger.error(f"ArXiv source validation failed: {e}")
            return False
