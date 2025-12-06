"""Review stage - reviews and improves the article draft."""

from dataclasses import dataclass
from typing import Optional

from ..llm import Message
from ..models import Article, ArticleSection, CodeExample
from .base import PipelineStage, StageContext


@dataclass
class ReviewResult:
    """Result of a review iteration."""

    article: Article
    score: float
    needs_improvement: bool
    feedback: dict
    iteration: int


class ReviewStage(PipelineStage[Article, Article]):
    """Stage for reviewing and improving article draft."""

    name = "review"
    description = "Reviewing and improving article"

    async def execute(
        self,
        input_data: Article,
        context: StageContext,
    ) -> Article:
        """Review and improve article with iterative refinement.

        Args:
            input_data: Article draft
            context: Pipeline context

        Returns:
            Improved article
        """
        # Get iteration settings from config
        pipeline_config = context.config.get("pipeline", {})
        max_revisions = pipeline_config.get("max_revisions", 3)
        score_threshold = pipeline_config.get("revision_threshold", 8.0)

        current_article = input_data
        all_feedback: list[dict] = []

        for iteration in range(max_revisions):
            # Generate review feedback
            review_result = await self._generate_review(current_article, context)
            all_feedback.append(review_result)

            score = float(review_result.get("overall_score", 0))
            needs_improvement = review_result.get("needs_improvement", False)

            # Save current iteration feedback
            context.save_artifact(f"review_feedback_{iteration}", review_result)

            # Check if we've reached the quality threshold
            if score >= score_threshold or not needs_improvement:
                context.save_artifact("review_feedback", review_result)
                context.save_artifact("review_iterations", iteration + 1)
                context.save_artifact("final_score", score)
                return current_article

            # Apply improvements
            current_article = await self._apply_improvements(
                current_article, review_result, context
            )

        # Save final feedback after all iterations
        context.save_artifact("review_feedback", all_feedback[-1] if all_feedback else {})
        context.save_artifact("review_iterations", max_revisions)
        context.save_artifact("final_score", float(all_feedback[-1].get("overall_score", 0)) if all_feedback else 0)

        return current_article

    async def _generate_review(
        self,
        article: Article,
        context: StageContext,
    ) -> dict:
        """Generate review feedback for the article.

        Args:
            article: Article to review
            context: Pipeline context

        Returns:
            Review feedback dict
        """
        # Convert article to markdown for review
        article_content = article.to_markdown()

        prompt = context.prompt_loader.load(
            "review",
            article_title=article.frontmatter.title,
            article_content=article_content,
        )

        messages = [
            Message(role="system", content=self._get_system_prompt()),
            Message(role="user", content=prompt),
        ]

        return await context.llm_client.complete_json(messages)

    async def _apply_improvements(
        self,
        article: Article,
        review: dict,
        context: StageContext,
    ) -> Article:
        """Apply review improvements to article.

        Args:
            article: Original article
            review: Review feedback
            context: Pipeline context

        Returns:
            Improved article
        """
        improvements = review.get("improvements", [])

        # For now, just update sections that need improvement
        improved_sections = []

        for i, section in enumerate(article.sections):
            section_improvements = [
                imp for imp in improvements
                if imp.get("section_index") == i
            ]

            if section_improvements:
                # Regenerate this section with improvements
                improved_section = await self._improve_section(
                    section, section_improvements, context
                )
                improved_sections.append(improved_section)
            else:
                improved_sections.append(section)

        return Article(
            frontmatter=article.frontmatter,
            sections=improved_sections,
            references=article.references,
        )

    async def _improve_section(
        self,
        section: ArticleSection,
        improvements: list[dict],
        context: StageContext,
    ) -> ArticleSection:
        """Improve a single section based on feedback.

        Args:
            section: Section to improve
            improvements: List of improvements for this section
            context: Pipeline context

        Returns:
            Improved section
        """
        improvement_text = "\n".join(
            f"- {imp.get('suggestion', '')}"
            for imp in improvements
        )

        prompt = f"""以下のセクションを改善してください。

## 現在のセクション
見出し: {section.heading}
内容:
{section.content}

## 改善点
{improvement_text}

## タスク
上記の改善点を反映した新しいセクション内容を生成してください。

JSON形式で回答してください：
```json
{{
  "content": "改善されたセクション内容"
}}
```
"""

        messages = [
            Message(
                role="system",
                content="技術記事の改善を行うアシスタントです。指摘された改善点を反映して、より良い記事にしてください。",
            ),
            Message(role="user", content=prompt),
        ]

        result = await context.llm_client.complete_json(messages)

        return ArticleSection(
            heading=section.heading,
            level=section.level,
            content=result.get("content", section.content),
            code_examples=section.code_examples,
        )

    def validate_input(self, input_data: Article) -> bool:
        """Validate that we have a valid article."""
        return isinstance(input_data, Article) and bool(input_data.sections)

    def _get_system_prompt(self) -> str:
        """Get system prompt for review."""
        return """あなたは技術記事の編集者です。
与えられた記事をレビューし、改善点を指摘してください。

以下の観点でレビューしてください：
1. 技術的正確性 - 内容は正確か
2. 可読性 - 初心者にもわかりやすいか
3. 構成 - 論理的な流れになっているか
4. コードの品質 - コードは動作するか、ベストプラクティスに従っているか
5. 実用性 - 読者が実践できる内容か

必ずJSON形式で回答してください。"""
