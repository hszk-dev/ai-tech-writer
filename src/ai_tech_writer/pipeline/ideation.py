"""Ideation stage - generates article ideas from topics."""

import json
from typing import Any

from ..llm import Message
from ..models import ArticleIdea
from ..web import TavilySearchProvider, create_search_provider
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

        # Search for related information
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
        """Search for information about the topic.

        Args:
            topic: Topic to search for
            context: Pipeline context

        Returns:
            List of search results
        """
        web_config = context.config.get("web_search", {})

        try:
            provider = create_search_provider(web_config)
            results = await provider.search(
                query=f"{topic} 技術記事 チュートリアル",
                num_results=web_config.get("num_results", 10),
            )
            return [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                }
                for r in results
            ]
        except Exception as e:
            # Log error but continue without search results
            print(f"Web search failed: {e}")
            return []

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
