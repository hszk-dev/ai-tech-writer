"""Web search providers for gathering information."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    content: Optional[str] = None  # Full page content if available
    score: float = 0.0


class WebSearchProvider(ABC):
    """Abstract base class for web search providers."""

    @abstractmethod
    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        """Search the web for a query.

        Args:
            query: Search query
            num_results: Maximum number of results

        Returns:
            List of search results
        """
        ...


class TavilySearchProvider(WebSearchProvider):
    """Tavily API search provider - optimized for AI agents."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        include_answer: bool = True,
        search_depth: str = "basic",
    ):
        """Initialize Tavily search provider.

        Args:
            api_key: Tavily API key (defaults to TAVILY_API_KEY env var)
            include_answer: Whether to include AI-generated answer
            search_depth: "basic" or "advanced"
        """
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key not provided. Set TAVILY_API_KEY environment variable."
            )

        self.include_answer = include_answer
        self.search_depth = search_depth
        self._client = None

    @property
    def client(self):
        """Get Tavily client, creating if needed."""
        if self._client is None:
            from tavily import TavilyClient

            self._client = TavilyClient(api_key=self.api_key)
        return self._client

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        """Search using Tavily API.

        Args:
            query: Search query
            num_results: Maximum number of results

        Returns:
            List of search results with content
        """
        # Tavily's Python client is synchronous, so we run it directly
        response = self.client.search(
            query=query,
            search_depth=self.search_depth,
            include_answer=self.include_answer,
            max_results=num_results,
        )

        results = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],  # Tavily returns full content
                    content=item.get("content"),
                    score=item.get("score", 0.0),
                )
            )

        return results

    async def search_with_answer(
        self,
        query: str,
        num_results: int = 10,
    ) -> tuple[str, list[SearchResult]]:
        """Search and get AI-generated answer.

        Args:
            query: Search query
            num_results: Maximum number of results

        Returns:
            Tuple of (answer, results)
        """
        response = self.client.search(
            query=query,
            search_depth=self.search_depth,
            include_answer=True,
            max_results=num_results,
        )

        answer = response.get("answer", "")
        results = []
        for item in response.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],
                    content=item.get("content"),
                    score=item.get("score", 0.0),
                )
            )

        return answer, results


class NoOpSearchProvider(WebSearchProvider):
    """No-op search provider for testing or when web search is disabled."""

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        """Return empty results.

        Args:
            query: Search query (ignored)
            num_results: Maximum number of results (ignored)

        Returns:
            Empty list
        """
        return []


def create_search_provider(config: dict) -> WebSearchProvider:
    """Create a search provider from configuration.

    Args:
        config: Web search configuration

    Returns:
        Configured search provider
    """
    provider = config.get("provider", "tavily")

    if provider == "tavily":
        return TavilySearchProvider(
            include_answer=config.get("include_answer", True),
            search_depth=config.get("search_depth", "basic"),
        )
    elif provider == "none":
        return NoOpSearchProvider()
    else:
        raise ValueError(f"Unknown search provider: {provider}")
