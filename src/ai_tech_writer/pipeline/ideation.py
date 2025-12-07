"""Ideation stage - generates article ideas from topics."""

import json
from typing import Any

from ..llm import Message
from ..models import ArticleIdea
from ..web import QueryOptimizer, TavilySearchProvider, create_search_provider
from .base import PipelineStage, StageContext


class IdeationStage(PipelineStage[dict[str, str], ArticleIdea]):
    """Stage for generating article ideas from topics."""

    name = "ideation"
    description = "Generating article idea and structure"

    async def execute(
        self,
        input_data: dict[str, str],
        context: StageContext,
    ) -> ArticleIdea:
        """Generate article idea from topic.

        Args:
            input_data: Dict with "topic" key
            context: Pipeline context

        Returns:
            ArticleIdea with title, sections, etc.
        """
        topic = input_data.get("topic", "")

        # Search for related information (with optimized queries)
        search_results = await self._search_topic(topic, context)

        # Save search results as artifact for later inspection
        context.save_artifact("search_results", search_results)

        # Generate idea using LLM
        prompt = context.prompt_loader.load(
            "ideation",
            topic=topic,
            search_results=self._format_search_results(search_results),
        )

        messages = [
            Message(role="system", content=self._get_system_prompt()),
            Message(role="user", content=prompt),
        ]

        result = await context.llm_client.complete_json(messages)

        return ArticleIdea(
            title=result.get("title", topic),
            emoji=result.get("emoji", "📝"),
            topics=result.get("topics", [])[:5],
            target_audience=result.get("target_audience", ""),
            problem_to_solve=result.get("problem_to_solve", ""),
            key_takeaways=result.get("key_takeaways", []),
            suggested_sections=result.get("suggested_sections", []),
        )

    def validate_input(self, input_data: dict[str, str]) -> bool:
        """Validate that topic is provided."""
        return isinstance(input_data, dict) and bool(input_data.get("topic"))

    async def _search_topic(
        self,
        topic: str,
        context: StageContext,
    ) -> list[dict[str, Any]]:
        """Search for information about the topic using optimized queries.

        Args:
            topic: Topic to search for
            context: Pipeline context

        Returns:
            List of search results
        """
        web_config = context.config.get("web_search", {})

        # Skip if provider is "none"
        if web_config.get("provider") == "none":
            return []

        try:
            # Use query optimization if enabled
            optimize_queries = web_config.get("optimize_queries", True)
            if optimize_queries:
                queries = await self._get_optimized_queries(topic, context)
                context.save_artifact("search_queries", queries)
            else:
                queries = [f"{topic} 技術記事 チュートリアル"]

            # Create search provider with caching
            enable_cache = web_config.get("enable_cache", True)
            provider = create_search_provider(
                web_config,
                enable_cache=enable_cache,
                cache_dir=context.working_dir / ".cache" / "search",
            )

            # Search with multiple queries and combine results
            all_results: list[dict[str, Any]] = []
            seen_urls: set[str] = set()
            num_results = web_config.get("num_results", 10)

            for query in queries:
                results = await provider.search(
                    query=query,
                    num_results=num_results // len(queries) + 2,  # Get a few extra per query
                )
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_results.append({
                            "title": r.title,
                            "url": r.url,
                            "snippet": r.snippet,
                            "query": query,
                        })

            # Sort by score and limit results
            return all_results[:num_results]

        except Exception as e:
            # Log error but continue without search results
            print(f"Web search failed: {e}")
            return []

    async def _get_optimized_queries(
        self,
        topic: str,
        context: StageContext,
    ) -> list[str]:
        """Get optimized search queries for the topic.

        Args:
            topic: Original topic
            context: Pipeline context

        Returns:
            List of optimized queries
        """
        try:
            optimizer = QueryOptimizer(context.llm_client)
            return await optimizer.optimize_query(topic)
        except Exception as e:
            # Fall back to simple query if optimization fails
            print(f"Query optimization failed: {e}")
            return [f"{topic} 技術記事 チュートリアル"]

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results for prompt.

        Args:
            results: Search results

        Returns:
            Formatted string
        """
        if not results:
            return "（検索結果なし）"

        formatted = []
        for i, r in enumerate(results[:5], 1):
            formatted.append(f"{i}. {r['title']}")
            formatted.append(f"   URL: {r['url']}")
            formatted.append(f"   概要: {r['snippet'][:200]}...")
            formatted.append("")

        return "\n".join(formatted)

    def _get_system_prompt(self) -> str:
        """Get system prompt for ideation."""
        return """あなたは技術記事のアイデア生成を専門とするAIアシスタントです。
与えられたトピックについて、Zenn/Qiita向けの技術記事のアイデアを生成してください。

以下の点を考慮してください：
- ターゲット読者は誰か（初心者、中級者、上級者）
- どのような問題を解決する記事か
- 読者が得られる具体的な知識やスキル
- 実践的なコード例を含められるか

必ずJSON形式で回答してください。"""
