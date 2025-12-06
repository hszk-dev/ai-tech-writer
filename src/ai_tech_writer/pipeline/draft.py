"""Draft stage - writes full article from outline."""

from ..llm import Message
from ..models import (
    Article,
    ArticleFrontmatter,
    ArticleOutline,
    ArticleSection,
    ArticleType,
    CodeExample,
)
from .base import PipelineStage, StageContext


class DraftStage(PipelineStage[ArticleOutline, Article]):
    """Stage for writing the full article draft."""

    name = "draft"
    description = "Writing article draft"

    async def execute(
        self,
        input_data: ArticleOutline,
        context: StageContext,
    ) -> Article:
        """Write full article from outline.

        Args:
            input_data: Article outline
            context: Pipeline context

        Returns:
            Complete article draft
        """
        idea = input_data.idea

        # Create frontmatter
        frontmatter = ArticleFrontmatter(
            title=idea.title,
            emoji=idea.emoji,
            type=ArticleType.TECH,
            topics=idea.topics[:5],
        )

        # Generate each section
        sections = []

        # Introduction
        intro_content = await self._generate_section(
            context=context,
            section_title="はじめに",
            section_context=input_data.introduction,
            outline=input_data,
            is_intro=True,
        )
        sections.append(
            ArticleSection(
                heading="はじめに",
                level=2,
                content=intro_content["content"],
                code_examples=[],
            )
        )

        # Main sections
        for outline_section in input_data.sections:
            section_result = await self._generate_section(
                context=context,
                section_title=outline_section.heading,
                section_context="\n".join(
                    f"- {p}" for p in outline_section.key_points
                ),
                outline=input_data,
                needs_code=outline_section.code_needed,
            )

            code_examples = []
            if section_result.get("code_examples"):
                for code in section_result["code_examples"]:
                    code_examples.append(
                        CodeExample(
                            language=code.get("language", "python"),
                            code=code.get("code", ""),
                            description=code.get("description", ""),
                        )
                    )

            sections.append(
                ArticleSection(
                    heading=outline_section.heading,
                    level=2,
                    content=section_result["content"],
                    code_examples=code_examples,
                )
            )

        # Conclusion
        conclusion_content = await self._generate_section(
            context=context,
            section_title="まとめ",
            section_context="\n".join(
                f"- {p}" for p in input_data.conclusion_points
            ),
            outline=input_data,
            is_conclusion=True,
        )
        sections.append(
            ArticleSection(
                heading="まとめ",
                level=2,
                content=conclusion_content["content"],
                code_examples=[],
            )
        )

        return Article(
            frontmatter=frontmatter,
            sections=sections,
            references=[],
        )

    async def _generate_section(
        self,
        context: StageContext,
        section_title: str,
        section_context: str,
        outline: ArticleOutline,
        is_intro: bool = False,
        is_conclusion: bool = False,
        needs_code: bool = False,
    ) -> dict:
        """Generate a single section.

        Args:
            context: Pipeline context
            section_title: Title of the section
            section_context: Context/key points for the section
            outline: Full article outline
            is_intro: Whether this is the introduction
            is_conclusion: Whether this is the conclusion
            needs_code: Whether code examples are needed

        Returns:
            Dict with content and optional code_examples
        """
        prompt = context.prompt_loader.load(
            "draft",
            article_title=outline.idea.title,
            target_audience=outline.idea.target_audience,
            section_title=section_title,
            section_context=section_context,
            is_intro=is_intro,
            is_conclusion=is_conclusion,
            needs_code=needs_code,
        )

        messages = [
            Message(role="system", content=self._get_system_prompt()),
            Message(role="user", content=prompt),
        ]

        return await context.llm_client.complete_json(messages)

    def validate_input(self, input_data: ArticleOutline) -> bool:
        """Validate that we have a valid outline."""
        return isinstance(input_data, ArticleOutline) and bool(input_data.sections)

    def _get_system_prompt(self) -> str:
        """Get system prompt for draft generation."""
        return """あなたは技術記事を執筆する専門家です。
与えられたセクション情報から、読みやすく実用的な記事コンテンツを生成してください。

以下の点を心がけてください：
- 具体的な説明と実例を含める
- 専門用語は初出時に説明する
- コードは実際に動作するものを書く
- 読者が実践できるように手順を明確に

必ずJSON形式で回答してください。"""
