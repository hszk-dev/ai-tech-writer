"""Pydantic schemas for LLM structured outputs."""

from pydantic import BaseModel, Field

# ============================================================
# Ideation Stage Schemas
# ============================================================


class ArticleIdeaOutput(BaseModel):
    """LLM output schema for article idea generation."""

    title: str = Field(description="記事タイトル")
    emoji: str = Field(description="絵文字1つ", default="📝")
    topics: list[str] = Field(description="タグ（最大5つ）")
    target_audience: str = Field(description="想定読者")
    problem_to_solve: str = Field(description="解決する課題")
    key_takeaways: list[str] = Field(default_factory=list, description="学べること")
    suggested_sections: list[str] = Field(default_factory=list, description="提案セクション")
    novelty_points: list[str] = Field(default_factory=list, description="新規性ポイント")
    buzz_factors: list[str] = Field(default_factory=list, description="バズる理由")
    trend_relevance: str = Field(default="medium", description="トレンド関連性")
    hook_elements: list[str] = Field(default_factory=list, description="フック要素")
    code_references: list[dict[str, str]] = Field(default_factory=list, description="コード参照")


# ============================================================
# Outline Stage Schemas
# ============================================================


class OutlineSectionOutput(BaseModel):
    """Single section in outline output."""

    heading: str = Field(description="セクションの見出し")
    key_points: list[str] = Field(default_factory=list, description="ポイント")
    code_needed: bool = Field(default=False, description="コード例が必要か")
    estimated_words: int = Field(default=200, ge=50, le=2000, description="推定文字数")


class ArticleOutlineOutput(BaseModel):
    """LLM output schema for article outline generation."""

    introduction: str = Field(description="はじめにの概要（2-3文）")
    sections: list[OutlineSectionOutput] = Field(description="セクションリスト")
    conclusion_points: list[str] = Field(default_factory=list, description="まとめポイント")


# ============================================================
# Draft Stage Schemas
# ============================================================


class CodeExampleOutput(BaseModel):
    """Code example in draft output."""

    language: str = Field(description="プログラミング言語")
    code: str = Field(description="コード本体")
    description: str = Field(default="", description="コードの説明")


class SectionContentOutput(BaseModel):
    """LLM output schema for section content (without code)."""

    content: str = Field(description="セクションの本文")


class SectionWithCodeOutput(BaseModel):
    """LLM output schema for section content (with code examples)."""

    content: str = Field(description="セクションの本文")
    code_examples: list[CodeExampleOutput] = Field(
        default_factory=list, description="コード例のリスト"
    )


# ============================================================
# Review Stage Schemas
# ============================================================


class ImprovementItem(BaseModel):
    """Single improvement suggestion."""

    section_index: int = Field(description="0から始まるセクション番号")
    issue: str = Field(description="問題点の説明")
    suggestion: str = Field(description="改善の提案")


class ReviewOutput(BaseModel):
    """Review output schema."""

    overall_score: float = Field(ge=1, le=10, description="1-10のスコア")
    needs_improvement: bool = Field(description="改善が必要かどうか")
    summary: str = Field(description="全体的な評価コメント")
    strengths: list[str] = Field(description="良い点のリスト")
    improvements: list[ImprovementItem] = Field(default_factory=list, description="改善点のリスト")
    code_issues: list[ImprovementItem] = Field(
        default_factory=list, description="コードの問題点のリスト"
    )


class ImprovedSection(BaseModel):
    """Improved section output schema."""

    content: str = Field(description="改善されたセクション内容")


# ============================================================
# QueryOptimizer Schema
# ============================================================


class QueryOptimizationOutput(BaseModel):
    """LLM output schema for search query optimization."""

    queries: list[str] = Field(description="最適化された検索クエリ")
    reasoning: str = Field(default="", description="クエリ選定の理由")


# ============================================================
# Evaluator Schema (TreeSearch)
# ============================================================


class EvaluationOutput(BaseModel):
    """LLM output schema for article quality evaluation."""

    overall_score: float = Field(ge=0, le=10, description="総合スコア")
    structure_score: float = Field(ge=0, le=10, description="構造性スコア")
    accuracy_score: float = Field(ge=0, le=10, description="技術的正確性スコア")
    readability_score: float = Field(ge=0, le=10, description="可読性スコア")
    practicality_score: float = Field(ge=0, le=10, description="実用性スコア")
    novelty_score: float = Field(default=5.0, ge=0, le=10, description="新規性スコア")
    buzz_score: float = Field(default=5.0, ge=0, le=10, description="バズ可能性スコア")
    feedback: str = Field(default="", description="改善点コメント")
    buzz_improvement: str = Field(default="", description="バズるためのアドバイス")


# ============================================================
# TreeSearch Improvement Schema
# ============================================================


class ImprovedSectionItem(BaseModel):
    """Single improved section in tree search."""

    heading: str = Field(description="セクション名")
    content: str = Field(description="改善された内容")


class ImprovedDraftOutput(BaseModel):
    """LLM output schema for draft improvement in tree search."""

    sections: list[ImprovedSectionItem] = Field(
        default_factory=list, description="改善されたセクションのリスト"
    )


# ============================================================
# Pre-computed JSON Schemas for LLMClient.complete_json()
# ============================================================

IDEATION_SCHEMA = ArticleIdeaOutput.model_json_schema()
OUTLINE_SCHEMA = ArticleOutlineOutput.model_json_schema()
SECTION_SCHEMA = SectionContentOutput.model_json_schema()
SECTION_WITH_CODE_SCHEMA = SectionWithCodeOutput.model_json_schema()
REVIEW_SCHEMA = ReviewOutput.model_json_schema()
IMPROVE_SECTION_SCHEMA = ImprovedSection.model_json_schema()
QUERY_OPTIMIZATION_SCHEMA = QueryOptimizationOutput.model_json_schema()
EVALUATION_SCHEMA = EvaluationOutput.model_json_schema()
IMPROVED_DRAFT_SCHEMA = ImprovedDraftOutput.model_json_schema()
