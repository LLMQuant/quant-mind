"""ArXiv source implementation for paper acquisition."""

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import arxiv

from quantmind.models.paper import Paper
from quantmind.sources.base import BaseSource
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class ArxivSource(BaseSource):
    """ArXiv source for academic paper acquisition.

    Provides access to arXiv papers through the arXiv API with support
    for searching, filtering, and retrieving paper metadata.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ArXiv source.

        Args:
            config: Configuration dictionary with optional settings:
                - max_results: Maximum papers per query (default: 100)
                - sort_by: Sort criteria (default: arxiv.SortCriterion.SubmittedDate)
        """
        super().__init__(config)
        self.max_results = self.config.get("max_results", 100)
        self.sort_by = self.config.get(
            "sort_by", arxiv.SortCriterion.SubmittedDate
        )
        self.client = arxiv.Client()

    def search(self, query: str, max_results: int = 10) -> List[Paper]:
        """Search arXiv for papers matching the query.

        Args:
            query: Search query string (supports arXiv query syntax)
            max_results: Maximum number of results to return

        Returns:
            List of Paper objects
        """
        try:
            search = arxiv.Search(
                query=query, max_results=max_results, sort_by=self.sort_by
            )

            papers = []
            for result in self.client.results(search):
                paper = self._convert_arxiv_result(result)
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

    def get_recent(
        self, days: int = 7, categories: Optional[List[str]] = None
    ) -> List[Paper]:
        """Get recent papers from arXiv.

        Args:
            days: Number of days to look back
            categories: Optional list of arXiv categories to filter by

        Returns:
            List of recent Paper objects
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

        return self.search(query, max_results=self.max_results)

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
        categories = [cat.term for cat in result.categories]

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
