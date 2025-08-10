"""Search source for fetching content from search engines."""

from typing import List, Optional

from ddgs import DDGS
from quantmind.config import SearchSourceConfig
from quantmind.models.search import SearchContent
from quantmind.sources.base import BaseSource
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class SearchSource(BaseSource[SearchContent]):
    """
    SearchSource provides a way to fetch content from search engines.
    Currently, it uses DuckDuckGo as the search provider.
    """

    def __init__(self, config: Optional[SearchSourceConfig] = None):
        """
        Initializes the SearchSource with an optional configuration.

        Args:
            config: A SearchSourceConfig object. If not provided, a default config is used.
        """
        self.config = config or SearchSourceConfig()
        super().__init__(self.config)
        self.client = DDGS()

    def search(
        self, query: str, max_results: Optional[int] = None
    ) -> List[SearchContent]:
        """
        Performs a search query and returns a list of SearchContent objects.

        Args:
            query: The search query string.
            max_results: The maximum number of results to return. Defaults to the value in the config.

        Returns:
            A list of SearchContent objects.
        """
        if max_results is None:
            max_results = self.config.max_results

        try:
            results = self.client.text(
                query, max_results=max_results
            )
            search_content_list = [
                SearchContent(
                    title=result["title"],
                    url=result["href"],
                    snippet=result["body"],
                    query=query,
                    source=self.name,
                    meta_info={},
                )
                for result in results
            ]
            logger.info(f"Found {len(search_content_list)} results for query: '{query}'")
            return search_content_list
        except Exception as e:
            logger.error(f"An error occurred while searching with DuckDuckGo: {e}")
            return []

    def get_by_id(self, content_id: str) -> Optional[SearchContent]:
        """
        Retrieves content by its ID (URL). This is not a standard use case for a search source,
        but it's implemented for interface consistency. It performs a search for the URL.

        Args:
            content_id: The URL of the content to retrieve.

        Returns:
            A SearchContent object if the URL is found, otherwise None.
        """
        # A bit of a hack to satisfy the interface. Search for the URL.
        results = self.search(query=content_id, max_results=1)
        if results and results[0].url == content_id:
            return results[0]
        return None
