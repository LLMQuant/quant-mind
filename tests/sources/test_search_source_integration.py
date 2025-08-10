"""Integration tests for the SearchSource."""

import unittest

from quantmind.sources.search_source import SearchSource
from quantmind.config.sources import SearchSourceConfig
from quantmind.models.search import SearchContent


class TestSearchSourceIntegration(unittest.TestCase):
    """Test suite for the SearchSource with real network requests."""

    def setUp(self):
        """Set up the test case."""
        self.config = SearchSourceConfig(max_results=5)
        self.source = SearchSource(config=self.config)

    def test_search_finreport(self):
        """Test a real search for 'finreport'."""
        results = self.source.search("finreport")

        self.assertGreater(len(results), 0)
        self.assertIsInstance(results[0], SearchContent)
        self.assertIsNotNone(results[0].title)
        self.assertIsNotNone(results[0].url)
        self.assertIsNotNone(results[0].snippet)
        self.assertIn("finreport", results[0].query.lower())


if __name__ == "__main__":
    unittest.main()
