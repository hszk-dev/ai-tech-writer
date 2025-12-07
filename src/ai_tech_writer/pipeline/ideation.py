"""Ideation stage - generates article ideas from topics."""

from typing import Any, Union

from ..llm import Message
from ..llm.schemas import IDEATION_SCHEMA, ArticleIdeaOutput
from ..models import ArticleIdea
from ..web import QueryOptimizer, create_search_provider
from .base import PipelineStage, StageContext


class IdeationStage(PipelineStage[Union[dict[str, str], Any], ArticleIdea]):
    """Stage for generating article ideas from topics."""

    name = "ideation"
    description = "Generating article idea and structure"

    async def execute(
        self,
        input_data: Union[dict[str, str], Any],
        context: StageContext,
    ) -> ArticleIdea:
        """Generate article idea from topic.

        Args:
            input_data: Dict with "topic" key, or ProjectAnalysis from previous stage
            context: Pipeline context

        Returns:
            ArticleIdea with title, sections, etc.
        """
        # プロジェクト分析結果がある場合は専用処理
        if context.has_project_analysis():
            return await self._execute_with_project(input_data, context)

        # 従来のトピックベース処理
        topic = input_data.get("topic", "") if isinstance(input_data, dict) else ""

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

        result = await context.llm_client.complete_json(messages, schema=IDEATION_SCHEMA)
        validated = ArticleIdeaOutput.model_validate(result)

        return self._build_article_idea(validated, topic)

    async def _execute_with_project(
        self,
        input_data: Any,
        context: StageContext,
    ) -> ArticleIdea:
        """プロジェクト分析結果を使ったアイデア生成.

        Args:
            input_data: ProjectAnalysis or dict
            context: Pipeline context

        Returns:
            ArticleIdea with project-specific information
        """
        # トピックの取得（指定があれば使用）
        topic = ""
        if isinstance(input_data, dict):
            topic = input_data.get("topic", "")

        # プロジェクト分析結果をdict形式で取得
        project_analysis = context.project_analysis
        project_analysis_dict = project_analysis.to_dict() if project_analysis else {}

        # Web検索も併用
        search_results = []
        if topic:
            search_results = await self._search_topic(topic, context)
        context.save_artifact("search_results", search_results)

        # プロジェクトベースのプロンプトを使用
        prompt = context.prompt_loader.load(
            "project_ideation",
            project_analysis=project_analysis_dict,
            topic=topic if topic else None,
            search_results=self._format_search_results(search_results) if search_results else None,
        )

        messages = [
            Message(role="system", content=self._get_system_prompt_for_project()),
            Message(role="user", content=prompt),
        ]

        result = await context.llm_client.complete_json(messages, schema=IDEATION_SCHEMA)
        validated = ArticleIdeaOutput.model_validate(result)

        # プロジェクトコンテキストを含めてArticleIdeaを構築
        idea = self._build_article_idea(validated, topic or project_analysis.project_name)
        idea.project_context = project_analysis.get_summary() if project_analysis else None
        idea.code_references = validated.code_references

        return idea

    def _build_article_idea(self, validated: ArticleIdeaOutput, fallback_title: str) -> ArticleIdea:
        """バリデーション済みLLM結果からArticleIdeaを構築."""
        return ArticleIdea(
            title=validated.title or fallback_title,
            emoji=validated.emoji,
            topics=validated.topics[:5],
            target_audience=validated.target_audience,
            problem_to_solve=validated.problem_to_solve,
            key_takeaways=validated.key_takeaways,
            suggested_sections=validated.suggested_sections,
            novelty_points=validated.novelty_points,
            buzz_factors=validated.buzz_factors,
            trend_relevance=validated.trend_relevance,
            hook_elements=validated.hook_elements,
            code_references=validated.code_references,
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
                        all_results.append(
                            {
                                "title": r.title,
                                "url": r.url,
                                "snippet": r.snippet,
                                "query": query,
                            }
                        )

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
- **バズりそうなタイトルと切り口か**
- **既存記事にない新規性があるか**

必ずJSON形式で回答してください。"""

    def _get_system_prompt_for_project(self) -> str:
        """Get system prompt for project-based ideation."""
        return """あなたは技術記事のアイデア生成を専門とするAIアシスタントです。
プロジェクトの分析結果を元に、Zenn/Qiitaでバズりそうな技術記事のアイデアを生成してください。

以下の点を特に重視してください：
- **新規性**: 既存記事にない独自の視点・情報があるか
- **バズ可能性**: タイトルの引き、トレンド性、共有したくなる要素
- **実体験ベース**: プロジェクトの実際のコードに基づいた具体的な内容
- **実用性**: 読者がすぐに活用できる知見

プロジェクトの実際のコードを引用できることを最大限に活用してください。
必ずJSON形式で回答してください。"""
