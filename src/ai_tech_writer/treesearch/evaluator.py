"""Article quality evaluator for tree search."""

import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from ..llm import LLMClient, Message, PromptLoader
from ..llm.schemas import EVALUATION_SCHEMA, EvaluationOutput
from .node import ArticleState


@dataclass
class EvaluationResult:
    """Result of evaluating an article state."""

    overall_score: float
    structure_score: float
    accuracy_score: float
    readability_score: float
    practicality_score: float
    feedback: str
    novelty_score: float = 5.0
    buzz_score: float = 5.0
    buzz_improvement: str = ""
    node_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], node_id: Optional[str] = None) -> "EvaluationResult":
        """Create from dictionary.

        Args:
            data: Dictionary with evaluation data
            node_id: Optional node ID

        Returns:
            EvaluationResult instance
        """
        return cls(
            overall_score=float(data.get("overall_score", 0)),
            structure_score=float(data.get("structure_score", 0)),
            accuracy_score=float(data.get("accuracy_score", 0)),
            readability_score=float(data.get("readability_score", 0)),
            practicality_score=float(data.get("practicality_score", 0)),
            feedback=str(data.get("feedback", "")),
            novelty_score=float(data.get("novelty_score", 5.0)),
            buzz_score=float(data.get("buzz_score", 5.0)),
            buzz_improvement=str(data.get("buzz_improvement", "")),
            node_id=node_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "overall_score": self.overall_score,
            "structure_score": self.structure_score,
            "accuracy_score": self.accuracy_score,
            "readability_score": self.readability_score,
            "practicality_score": self.practicality_score,
            "feedback": self.feedback,
            "novelty_score": self.novelty_score,
            "buzz_score": self.buzz_score,
            "buzz_improvement": self.buzz_improvement,
            "node_id": self.node_id,
        }

    @classmethod
    def create_default(cls, node_id: Optional[str] = None) -> "EvaluationResult":
        """Create a default evaluation result with zero scores.

        Args:
            node_id: Optional node ID

        Returns:
            Default EvaluationResult
        """
        return cls(
            overall_score=0.0,
            structure_score=0.0,
            accuracy_score=0.0,
            readability_score=0.0,
            practicality_score=0.0,
            feedback="Evaluation not performed",
            novelty_score=0.0,
            buzz_score=0.0,
            buzz_improvement="",
            node_id=node_id,
        )


class ArticleEvaluator:
    """Evaluates article quality using LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
    ):
        """Initialize the evaluator.

        Args:
            llm_client: LLM client for completions
            prompt_loader: Prompt template loader
        """
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader

    async def evaluate(
        self,
        article_state: ArticleState,
        node_id: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate an article state.

        Args:
            article_state: The article state to evaluate
            node_id: Optional node ID for tracking

        Returns:
            EvaluationResult with scores and feedback
        """
        content = article_state.get_current_content()
        if not content:
            return EvaluationResult.create_default(node_id)

        # Load evaluation prompt
        prompt = self.prompt_loader.load("evaluate", content=content)

        messages = [
            Message(
                role="system",
                content="あなたは技術記事の品質評価者です。客観的かつ建設的な評価を行ってください。必ずJSON形式で回答してください。",
            ),
            Message(role="user", content=prompt),
        ]

        try:
            result = await self.llm_client.complete_json(messages, schema=EVALUATION_SCHEMA)
            validated = EvaluationOutput.model_validate(result)
            return EvaluationResult(
                overall_score=validated.overall_score,
                structure_score=validated.structure_score,
                accuracy_score=validated.accuracy_score,
                readability_score=validated.readability_score,
                practicality_score=validated.practicality_score,
                feedback=validated.feedback,
                novelty_score=validated.novelty_score,
                buzz_score=validated.buzz_score,
                buzz_improvement=validated.buzz_improvement,
                node_id=node_id,
            )
        except Exception as e:
            # Return default result on error
            return EvaluationResult(
                overall_score=0.0,
                structure_score=0.0,
                accuracy_score=0.0,
                readability_score=0.0,
                practicality_score=0.0,
                feedback=f"Evaluation failed: {str(e)}",
                novelty_score=0.0,
                buzz_score=0.0,
                buzz_improvement="",
                node_id=node_id,
            )

    async def evaluate_batch(
        self,
        states: list[tuple[ArticleState, str]],
        max_concurrent: int = 3,
    ) -> list[EvaluationResult]:
        """Evaluate multiple article states concurrently.

        Args:
            states: List of (ArticleState, node_id) tuples
            max_concurrent: Maximum concurrent evaluations

        Returns:
            List of EvaluationResults in same order as input
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def evaluate_with_semaphore(
            state: ArticleState,
            node_id: str,
        ) -> EvaluationResult:
            async with semaphore:
                return await self.evaluate(state, node_id)

        tasks = [evaluate_with_semaphore(state, node_id) for state, node_id in states]

        return await asyncio.gather(*tasks)

    def calculate_weighted_score(
        self,
        structure: float,
        accuracy: float,
        readability: float,
        practicality: float,
    ) -> float:
        """Calculate weighted overall score.

        Weights:
        - Structure: 20%
        - Accuracy: 30%
        - Readability: 25%
        - Practicality: 25%

        Args:
            structure: Structure score (0-10)
            accuracy: Accuracy score (0-10)
            readability: Readability score (0-10)
            practicality: Practicality score (0-10)

        Returns:
            Weighted overall score (0-10)
        """
        return structure * 0.20 + accuracy * 0.30 + readability * 0.25 + practicality * 0.25
