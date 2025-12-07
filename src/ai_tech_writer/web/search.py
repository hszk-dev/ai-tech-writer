"""Web search providers for gathering information."""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..llm import LLMClient

from ..llm.schemas import QUERY_OPTIMIZATION_SCHEMA, QueryOptimizationOutput


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    content: Optional[str] = None  # Full page content if available
    score: float = 0.0


@dataclass
class CacheEntry:
    """Cache entry for search results."""

    query: str
    results: list[dict]
    timestamp: str
    ttl_hours: int = 24


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


class CachedSearchProvider(WebSearchProvider):
    """Wrapper that adds caching to any search provider."""

    def __init__(
        self,
        provider: WebSearchProvider,
        cache_dir: Optional[Path] = None,
        ttl_hours: int = 24,
    ):
        """Initialize cached search provider.

        Args:
            provider: Underlying search provider
            cache_dir: Directory to store cache files
            ttl_hours: Cache time-to-live in hours
        """
        self.provider = provider
        self.cache_dir = cache_dir or Path(".cache/search")
        self.ttl_hours = ttl_hours
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str, num_results: int) -> str:
        """Generate cache key from query."""
        key_str = f"{query}:{num_results}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        return self.cache_dir / f"{cache_key}.json"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache is still valid."""
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, encoding="utf-8") as f:
                entry = json.load(f)
            timestamp = datetime.fromisoformat(entry["timestamp"])
            ttl = timedelta(hours=entry.get("ttl_hours", self.ttl_hours))
            return datetime.now() - timestamp < ttl
        except (json.JSONDecodeError, KeyError, ValueError):
            return False

    def _load_cache(self, cache_path: Path) -> list[SearchResult]:
        """Load results from cache."""
        with open(cache_path, encoding="utf-8") as f:
            entry = json.load(f)
        return [
            SearchResult(
                title=r["title"],
                url=r["url"],
                snippet=r["snippet"],
                content=r.get("content"),
                score=r.get("score", 0.0),
            )
            for r in entry["results"]
        ]

    def _save_cache(self, cache_path: Path, query: str, results: list[SearchResult]) -> None:
        """Save results to cache."""
        entry = CacheEntry(
            query=query,
            results=[asdict(r) for r in results],
            timestamp=datetime.now().isoformat(),
            ttl_hours=self.ttl_hours,
        )
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(asdict(entry), f, ensure_ascii=False, indent=2)

    async def search(
        self,
        query: str,
        num_results: int = 10,
    ) -> list[SearchResult]:
        """Search with caching.

        Args:
            query: Search query
            num_results: Maximum number of results

        Returns:
            List of search results (from cache or fresh)
        """
        cache_key = self._get_cache_key(query, num_results)
        cache_path = self._get_cache_path(cache_key)

        # Return cached results if valid
        if self._is_cache_valid(cache_path):
            return self._load_cache(cache_path)

        # Fetch fresh results
        results = await self.provider.search(query, num_results)

        # Cache the results
        self._save_cache(cache_path, query, results)

        return results


class QueryOptimizer:
    """Optimizes search queries using LLM."""

    def __init__(self, llm_client: "LLMClient"):
        """Initialize query optimizer.

        Args:
            llm_client: LLM client for query optimization
        """
        self.llm_client = llm_client

    async def optimize_query(self, topic: str, context: Optional[str] = None) -> list[str]:
        """Generate optimized search queries for a topic.

        Args:
            topic: Original topic
            context: Additional context (e.g., target audience)

        Returns:
            List of optimized search queries
        """
        from ..llm import Message

        context_text = f"\n追加コンテキスト: {context}" if context else ""

        prompt = f"""以下のトピックについて技術記事を書くための検索クエリを3つ生成してください。

トピック: {topic}{context_text}

検索クエリの生成ルール:
1. 日本語と英語の両方を考慮する
2. 具体的な技術用語を含める
3. 「チュートリアル」「入門」「ベストプラクティス」などの修飾語を適切に使う
4. 公式ドキュメントや信頼性の高い情報源を見つけやすいクエリにする

JSON形式で回答してください：
```json
{{
  "queries": [
    "検索クエリ1",
    "検索クエリ2",
    "検索クエリ3"
  ],
  "reasoning": "これらのクエリを選んだ理由"
}}
```
"""

        messages = [
            Message(
                role="system",
                content="あなたは効果的な検索クエリを生成する専門家です。技術記事を書くための情報収集に最適な検索クエリを提案してください。",
            ),
            Message(role="user", content=prompt),
        ]

        result = await self.llm_client.complete_json(messages, schema=QUERY_OPTIMIZATION_SCHEMA)
        validated = QueryOptimizationOutput.model_validate(result)
        return validated.queries if validated.queries else [f"{topic} 技術記事"]


def create_search_provider(
    config: dict,
    enable_cache: bool = True,
    cache_dir: Optional[Path] = None,
) -> WebSearchProvider:
    """Create a search provider from configuration.

    Args:
        config: Web search configuration
        enable_cache: Whether to enable caching
        cache_dir: Directory for cache files

    Returns:
        Configured search provider
    """
    provider_name = config.get("provider", "tavily")

    if provider_name == "tavily":
        provider = TavilySearchProvider(
            include_answer=config.get("include_answer", True),
            search_depth=config.get("search_depth", "basic"),
        )
    elif provider_name == "none":
        return NoOpSearchProvider()
    else:
        raise ValueError(f"Unknown search provider: {provider_name}")

    # Wrap with cache if enabled
    if enable_cache and provider_name != "none":
        cache_ttl = config.get("cache_ttl_hours", 24)
        provider = CachedSearchProvider(
            provider=provider,
            cache_dir=cache_dir or Path(".cache/search"),
            ttl_hours=cache_ttl,
        )

    return provider
