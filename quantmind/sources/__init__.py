"""Source API layer for content acquisition."""

from quantmind.sources.base import BaseSource

__all__ = ["BaseSource"]

# Conditionally import ArXiv source
try:
    from quantmind.sources.arxiv_source import ArxivSource

    __all__.append("ArxivSource")
except ImportError:
    pass

# Conditionally import Search source
try:
    from quantmind.sources.search_source import SearchSource

    __all__.append("SearchSource")
except ImportError:
    pass
