"""Web search module."""

from .search import (
    SearchResult,
    TavilySearchProvider,
    WebSearchProvider,
    create_search_provider,
)

__all__ = [
    "SearchResult",
    "TavilySearchProvider",
    "WebSearchProvider",
    "create_search_provider",
]
