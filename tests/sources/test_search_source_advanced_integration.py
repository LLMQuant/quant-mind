"""Integration tests for the advanced features of the SearchSource."""

import unittest

from quantmind.sources.search_source import SearchSource
from quantmind.config.sources import SearchSourceConfig
from quantmind.models.search import SearchContent


class TestSearchSourceAdvancedIntegration(unittest.TestCase):
    """Test suite for the advanced features of the SearchSource with real network requests."""

    def setUp(self):
        """Set up the test case."""
        self.source = SearchSource()

    def test_search_with_site_filter(self):
        """Test a real search with a site filter."""
        results = self.source.search("machine learning", site="arxiv.org")

        self.assertGreater(len(results), 0)
        for result in results:
            self.assertIn("arxiv.org", result.url)

    def test_search_with_filetype_filter(self):
        """Test a real search with a filetype filter."""
        results = self.source.search("financial report", filetype="pdf")

        self.assertGreater(len(results), 0)
        # We can't guarantee that all results will have a .pdf extension in the URL,
        # as the filetype search is a hint to the search engine.
        # However, we can check if the query was constructed correctly.
        self.assertIn("filetype:pdf", results[0].query)

    def test_search_with_date_filter(self):
        """Test a real search with a date filter."""
        results = self.source.search("AI", start_date="2023-01-01", end_date="2023-01-31")

        self.assertGreater(len(results), 0)
        self.assertIn("daterange:2023-01-01..2023-01-31", results[0].query)


if __name__ == "__main__":
    unittest.main()
