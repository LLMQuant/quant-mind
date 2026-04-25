"""Source API layer for content acquisition."""

from quantmind.sources.base import BaseSource

__all__ = ["BaseSource"]

# Conditionally import ArXiv source
try:
    from quantmind.sources.arxiv_source import ArxivSource  # noqa: F401

    __all__.append("ArxivSource")
except ImportError:
    pass
