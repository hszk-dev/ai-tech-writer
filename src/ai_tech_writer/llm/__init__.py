"""LLM client module."""

from .client import LLMClient, Message
from .prompts import PromptLoader
from .schemas import (
    EVALUATION_SCHEMA,
    IDEATION_SCHEMA,
    IMPROVE_SECTION_SCHEMA,
    IMPROVED_DRAFT_SCHEMA,
    OUTLINE_SCHEMA,
    QUERY_OPTIMIZATION_SCHEMA,
    REVIEW_SCHEMA,
    SECTION_SCHEMA,
    SECTION_WITH_CODE_SCHEMA,
    # Ideation
    ArticleIdeaOutput,
    # Outline
    ArticleOutlineOutput,
    # Draft
    CodeExampleOutput,
    # Evaluator
    EvaluationOutput,
    ImprovedDraftOutput,
    ImprovedSection,
    # TreeSearch
    ImprovedSectionItem,
    # Review
    ImprovementItem,
    OutlineSectionOutput,
    # QueryOptimizer
    QueryOptimizationOutput,
    ReviewOutput,
    SectionContentOutput,
    SectionWithCodeOutput,
)

__all__ = [
    "LLMClient",
    "Message",
    "PromptLoader",
    # Schemas
    "ArticleIdeaOutput",
    "IDEATION_SCHEMA",
    "ArticleOutlineOutput",
    "OutlineSectionOutput",
    "OUTLINE_SCHEMA",
    "CodeExampleOutput",
    "SectionContentOutput",
    "SectionWithCodeOutput",
    "SECTION_SCHEMA",
    "SECTION_WITH_CODE_SCHEMA",
    "ImprovementItem",
    "ReviewOutput",
    "ImprovedSection",
    "REVIEW_SCHEMA",
    "IMPROVE_SECTION_SCHEMA",
    "QueryOptimizationOutput",
    "QUERY_OPTIMIZATION_SCHEMA",
    "EvaluationOutput",
    "EVALUATION_SCHEMA",
    "ImprovedSectionItem",
    "ImprovedDraftOutput",
    "IMPROVED_DRAFT_SCHEMA",
]
