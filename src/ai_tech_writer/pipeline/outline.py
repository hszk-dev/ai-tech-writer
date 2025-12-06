"""Outline stage - creates detailed article outline from idea."""

from ..llm import Message
from ..models import ArticleIdea, ArticleOutline, OutlineSection
from .base import PipelineStage, StageContext


class OutlineStage(PipelineStage[ArticleIdea, ArticleOutline]):
    """Stage for creating detailed article outline."""

    name = "outline"
    description = "Creating detailed article outline"

    async def execute(
        self,
        input_data: ArticleIdea,
        context: StageContext,
    ) -> ArticleOutline:
        """Create detailed outline from article idea.

        Args:
            input_data: Article idea from ideation stage
            context: Pipeline context

        Returns:
            Detailed article outline
        """
        prompt = context.prompt_loader.load(
            "outline",
            title=input_data.title,
            target_audience=input_data.target_audience,
            problem_to_solve=input_data.problem_to_solve,
            key_takeaways="\n".join(f"- {t}" for t in input_data.key_takeaways),
            suggested_sections="\n".join(
                f"- {s}" for s in input_data.suggested_sections
            ),
        )

        messages = [
            Message(role="system", content=self._get_system_prompt()),
            Message(role="user", content=prompt),
        ]

        result = await context.llm_client.complete_json(messages)

        # Parse sections
        sections = []
        for s in result.get("sections", []):
            sections.append(
                OutlineSection(
                    heading=s.get("heading", ""),
                    key_points=s.get("key_points", []),
                    code_needed=s.get("code_needed", False),
                    estimated_words=s.get("estimated_words", 200),
                )
            )

        return ArticleOutline(
            idea=input_data,
            sections=sections,
            introduction=result.get("introduction", ""),
            conclusion_points=result.get("conclusion_points", []),
        )

    def validate_input(self, input_data: ArticleIdea) -> bool:
        """Validate that we have a valid article idea."""
        return isinstance(input_data, ArticleIdea) and bool(input_data.title)

    def _get_system_prompt(self) -> str:
        """Get system prompt for outline generation."""
        return """あなたは技術記事の構成を設計する専門家です。
与えられた記事アイデアから、詳細なアウトラインを作成してください。

各セクションについて：
- 具体的な見出し
- そのセクションで説明する重要ポイント
- コード例が必要かどうか
- 推定文字数

を含めてください。必ずJSON形式で回答してください。"""
