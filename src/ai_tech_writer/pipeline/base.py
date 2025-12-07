"""Base classes for pipeline stages."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

from ..llm import LLMClient, PromptLoader

if TYPE_CHECKING:
    from ..analysis import ProjectAnalysis

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


@dataclass
class StageContext:
    """Shared context passed through pipeline stages."""

    llm_client: LLMClient
    prompt_loader: PromptLoader
    config: dict[str, Any]
    working_dir: Path
    artifacts: dict[str, Any] = field(default_factory=dict)
    # プロジェクト分析関連
    project_analysis: Optional["ProjectAnalysis"] = None
    project_path: Optional[Path] = None

    def save_artifact(self, name: str, data: Any) -> None:
        """Save an artifact for later stages."""
        self.artifacts[name] = data

    def get_artifact(self, name: str) -> Any:
        """Get a saved artifact."""
        return self.artifacts.get(name)

    def has_project_analysis(self) -> bool:
        """Check if project analysis is available."""
        return self.project_analysis is not None


class PipelineStage(ABC, Generic[InputT, OutputT]):
    """Base class for pipeline stages."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    async def execute(
        self,
        input_data: InputT,
        context: StageContext,
    ) -> OutputT:
        """Execute this stage.

        Args:
            input_data: Input from previous stage
            context: Shared pipeline context

        Returns:
            Output for next stage
        """
        ...

    def validate_input(self, input_data: InputT) -> bool:
        """Validate input before execution.

        Args:
            input_data: Input to validate

        Returns:
            True if valid, False otherwise
        """
        return input_data is not None
