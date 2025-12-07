"""Pipeline orchestrator for article generation."""

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel

from ..llm import LLMClient, PromptLoader
from ..models import Article, ArticleIdea
from .base import PipelineStage, StageContext

console = Console()


def _serialize_output(obj: Any) -> Any:
    """Serialize dataclass or other objects to dict for JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for key, value in asdict(obj).items():
            result[key] = _serialize_output(value)
        return result
    elif isinstance(obj, list):
        return [_serialize_output(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_output(v) for k, v in obj.items()}
    elif hasattr(obj, "value"):  # Enum
        return obj.value
    return obj


class ArticlePipeline:
    """Orchestrates the article generation pipeline."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_loader: PromptLoader,
        config: dict[str, Any],
        working_dir: Optional[Path] = None,
        verbose: bool = False,
    ):
        """Initialize pipeline.

        Args:
            llm_client: LLM client for completions
            prompt_loader: Prompt template loader
            config: Pipeline configuration
            working_dir: Working directory for outputs
            verbose: Whether to show detailed output
        """
        self.llm_client = llm_client
        self.prompt_loader = prompt_loader
        self.config = config
        self.working_dir = working_dir or Path("outputs")
        self.stages: list[PipelineStage] = []
        self.verbose = verbose

    def add_stage(self, stage: PipelineStage) -> "ArticlePipeline":
        """Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add

        Returns:
            Self for chaining
        """
        self.stages.append(stage)
        return self

    async def run(self, topic: str) -> Article:
        """Execute the full pipeline.

        Args:
            topic: Initial topic for article generation

        Returns:
            Generated article

        Raises:
            ValueError: If pipeline has no stages
            RuntimeError: If a stage fails
        """
        if not self.stages:
            raise ValueError("Pipeline has no stages")

        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Create intermediate directory for verbose output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_dir = self.working_dir / f".intermediate_{timestamp}"
        if self.verbose or self.config.get("pipeline", {}).get("save_intermediate"):
            intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Create context
        context = StageContext(
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
            config=self.config,
            working_dir=self.working_dir,
        )

        current_input: Any = {"topic": topic}

        for i, stage in enumerate(self.stages):
            console.print(
                Panel(
                    f"[bold blue]{stage.name}[/bold blue]\n{stage.description}",
                    title=f"Stage {i + 1}/{len(self.stages)}",
                )
            )

            if not stage.validate_input(current_input):
                raise RuntimeError(f"Invalid input for stage {stage.name}")

            try:
                output = await stage.execute(current_input, context)
                context.save_artifact(stage.name, output)
                current_input = output

                # Show completion with extra info for review stage
                if stage.name == "review":
                    iterations = context.get_artifact("review_iterations") or 1
                    final_score = context.get_artifact("final_score") or 0
                    console.print(
                        f"[green]✓[/green] {stage.name} completed "
                        f"[dim](iterations: {iterations}, score: {final_score:.1f})[/dim]"
                    )
                else:
                    console.print(f"[green]✓[/green] {stage.name} completed")

                # Save and display intermediate output
                if self.verbose or self.config.get("pipeline", {}).get("save_intermediate"):
                    self._save_intermediate(stage.name, output, intermediate_dir)

                    # Save search results after ideation stage
                    if stage.name == "ideation":
                        search_results = context.get_artifact("search_results")
                        if search_results:
                            self._save_intermediate(
                                "search_results", search_results, intermediate_dir
                            )

                    # Save review feedback after review stage (all iterations)
                    if stage.name == "review":
                        # Save code validation and execution results
                        code_validation = context.get_artifact("code_validation")
                        if code_validation:
                            self._save_intermediate(
                                "code_validation", code_validation, intermediate_dir
                            )
                        code_execution = context.get_artifact("code_execution")
                        if code_execution:
                            self._save_intermediate(
                                "code_execution", code_execution, intermediate_dir
                            )

                        # Save each iteration's feedback
                        max_revisions = self.config.get("pipeline", {}).get("max_revisions", 3)
                        for i in range(max_revisions):
                            feedback = context.get_artifact(f"review_feedback_{i}")
                            if feedback:
                                self._save_intermediate(
                                    f"review_feedback_{i}", feedback, intermediate_dir
                                )
                        # Save final feedback
                        review_feedback = context.get_artifact("review_feedback")
                        if review_feedback:
                            self._save_intermediate(
                                "review_feedback", review_feedback, intermediate_dir
                            )

                if self.verbose:
                    if stage.name == "review":
                        # For review, display feedback from context (not the article output)
                        self._display_review_output(context)
                    else:
                        self._display_stage_output(stage.name, output)

            except Exception as e:
                console.print(f"[red]✗[/red] {stage.name} failed: {e}")
                raise RuntimeError(f"Stage {stage.name} failed: {e}") from e

        if self.verbose or self.config.get("pipeline", {}).get("save_intermediate"):
            console.print(f"\n[dim]Intermediate files saved to: {intermediate_dir}[/dim]")

        # Extract final article
        return self._build_article(context)

    async def run_with_project(
        self,
        project_path: str,
        topic: Optional[str] = None,
    ) -> Article:
        """Execute the pipeline with project analysis.

        Args:
            project_path: Path to the project to analyze
            topic: Optional topic to focus on (if not provided, uses project analysis results)

        Returns:
            Generated article

        Raises:
            ValueError: If pipeline has no stages
            RuntimeError: If a stage fails
        """
        if not self.stages:
            raise ValueError("Pipeline has no stages")

        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Create intermediate directory for verbose output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_dir = self.working_dir / f".intermediate_{timestamp}"
        if self.verbose or self.config.get("pipeline", {}).get("save_intermediate"):
            intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Create context
        context = StageContext(
            llm_client=self.llm_client,
            prompt_loader=self.prompt_loader,
            config=self.config,
            working_dir=self.working_dir,
        )

        # Start with project path and optional topic
        current_input: Any = {
            "project_path": project_path,
            "topic": topic or "",
        }

        for i, stage in enumerate(self.stages):
            console.print(
                Panel(
                    f"[bold blue]{stage.name}[/bold blue]\n{stage.description}",
                    title=f"Stage {i + 1}/{len(self.stages)}",
                )
            )

            # ProjectAnalysisStage以外は入力検証をスキップ（入力形式が異なるため）
            if stage.name != "project_analysis" and hasattr(current_input, "to_dict"):
                # ProjectAnalysisの出力はそのまま渡す
                pass
            elif stage.name != "project_analysis" and not stage.validate_input(current_input):
                # 通常のステージでdict以外が来た場合は検証スキップ
                pass

            try:
                output = await stage.execute(current_input, context)
                context.save_artifact(stage.name, output)
                current_input = output

                # Show completion with extra info for review stage
                if stage.name == "review":
                    iterations = context.get_artifact("review_iterations") or 1
                    final_score = context.get_artifact("final_score") or 0
                    console.print(
                        f"[green]✓[/green] {stage.name} completed "
                        f"[dim](iterations: {iterations}, score: {final_score:.1f})[/dim]"
                    )
                elif stage.name == "project_analysis":
                    analysis = output
                    idea_count = 0
                    if hasattr(analysis, "article_ideas"):
                        idea_count = len(analysis.article_ideas)
                    console.print(
                        f"[green]✓[/green] {stage.name} completed [dim](ideas: {idea_count})[/dim]"
                    )
                else:
                    console.print(f"[green]✓[/green] {stage.name} completed")

                # Save and display intermediate output
                if self.verbose or self.config.get("pipeline", {}).get("save_intermediate"):
                    self._save_intermediate(stage.name, output, intermediate_dir)

                    # Save project analysis details
                    if stage.name == "project_analysis":
                        project_analysis = context.get_artifact("project_analysis")
                        if project_analysis:
                            self._save_intermediate(
                                "project_analysis_details", project_analysis, intermediate_dir
                            )

                    # Save search results after ideation stage
                    if stage.name == "ideation":
                        search_results = context.get_artifact("search_results")
                        if search_results:
                            self._save_intermediate(
                                "search_results", search_results, intermediate_dir
                            )

                    # Save review feedback after review stage
                    if stage.name == "review":
                        code_validation = context.get_artifact("code_validation")
                        if code_validation:
                            self._save_intermediate(
                                "code_validation", code_validation, intermediate_dir
                            )
                        code_execution = context.get_artifact("code_execution")
                        if code_execution:
                            self._save_intermediate(
                                "code_execution", code_execution, intermediate_dir
                            )

                        max_revisions = self.config.get("pipeline", {}).get("max_revisions", 3)
                        for idx in range(max_revisions):
                            feedback = context.get_artifact(f"review_feedback_{idx}")
                            if feedback:
                                self._save_intermediate(
                                    f"review_feedback_{idx}", feedback, intermediate_dir
                                )
                        review_feedback = context.get_artifact("review_feedback")
                        if review_feedback:
                            self._save_intermediate(
                                "review_feedback", review_feedback, intermediate_dir
                            )

                if self.verbose:
                    if stage.name == "review":
                        self._display_review_output(context)
                    elif stage.name == "project_analysis":
                        self._display_project_analysis_output(output)
                    else:
                        self._display_stage_output(stage.name, output)

            except Exception as e:
                console.print(f"[red]✗[/red] {stage.name} failed: {e}")
                raise RuntimeError(f"Stage {stage.name} failed: {e}") from e

        if self.verbose or self.config.get("pipeline", {}).get("save_intermediate"):
            console.print(f"\n[dim]Intermediate files saved to: {intermediate_dir}[/dim]")

        # Extract final article
        return self._build_article(context)

    def _display_project_analysis_output(self, output: Any) -> None:
        """Display project analysis output."""
        try:
            if hasattr(output, "project_name"):
                ideas_text = (
                    "\n".join(
                        f"  • [{idea.buzz_potential.upper()}] {idea.title}"
                        for idea in output.article_ideas[:3]
                    )
                    if hasattr(output, "article_ideas")
                    else "  (なし)"
                )

                console.print(
                    Panel(
                        f"[bold]Project:[/bold] {output.project_name}\n"
                        f"[bold]Tech Stack:[/bold] {', '.join(output.tech_stack.languages)}\n"
                        f"[bold]Architecture:[/bold] {output.architecture.pattern}\n\n"
                        f"[bold]Article Ideas:[/bold]\n{ideas_text}",
                        title="[cyan]Project Analysis Output[/cyan]",
                        border_style="cyan",
                    )
                )
        except Exception as e:
            console.print(f"[yellow]Could not display project analysis output: {e}[/yellow]")

    def _save_intermediate(self, stage_name: str, output: Any, output_dir: Path) -> None:
        """Save intermediate stage output to file."""
        try:
            serialized = _serialize_output(output)
            output_file = output_dir / f"{stage_name}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(serialized, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save {stage_name} output: {e}[/yellow]")

    def _display_stage_output(self, stage_name: str, output: Any) -> None:
        """Display stage output in verbose mode."""
        try:
            serialized = _serialize_output(output)

            if stage_name == "ideation":
                idea = serialized
                console.print(
                    Panel(
                        f"[bold]Title:[/bold] {idea.get('title', 'N/A')}\n"
                        f"[bold]Emoji:[/bold] {idea.get('emoji', 'N/A')}\n"
                        f"[bold]Topics:[/bold] {', '.join(idea.get('topics', []))}\n"
                        f"[bold]Target:[/bold] {idea.get('target_audience', 'N/A')}\n"
                        f"[bold]Problem:[/bold] {idea.get('problem_to_solve', 'N/A')}\n"
                        f"[bold]Sections:[/bold] {', '.join(idea.get('suggested_sections', []))}",
                        title="[cyan]Ideation Output[/cyan]",
                        border_style="cyan",
                    )
                )

            elif stage_name == "outline":
                outline = serialized
                sections_text = "\n".join(
                    f"  • {s.get('heading', 'N/A')} (code: {s.get('code_needed', False)})"
                    for s in outline.get("sections", [])
                )
                intro = outline.get("introduction", "N/A")[:200]
                conclusion_points = outline.get("conclusion_points", [])
                conclusion_text = "\n".join(f"  • {p}" for p in conclusion_points)
                console.print(
                    Panel(
                        f"[bold]Introduction:[/bold]\n{intro}...\n\n"
                        f"[bold]Sections:[/bold]\n{sections_text}\n\n"
                        f"[bold]Conclusion Points:[/bold]\n{conclusion_text}",
                        title="[cyan]Outline Output[/cyan]",
                        border_style="cyan",
                    )
                )

            elif stage_name == "draft":
                draft = serialized
                sections = draft.get("sections", [])
                title = draft.get("frontmatter", {}).get("title", "N/A")
                section_names = ", ".join(s.get("heading", "") for s in sections)
                console.print(
                    Panel(
                        f"[bold]Title:[/bold] {title}\n"
                        f"[bold]Sections:[/bold] {len(sections)}\n"
                        f"[bold]Section Names:[/bold] {section_names}",
                        title="[cyan]Draft Output[/cyan]",
                        border_style="cyan",
                    )
                )

            elif stage_name == "review":
                # Note: output here is the Article, review feedback is in context
                # This is handled by _display_review_output instead
                pass

        except Exception as e:
            console.print(f"[yellow]Could not display {stage_name} output: {e}[/yellow]")

    def _display_review_output(self, context: StageContext) -> None:
        """Display review output from context artifacts."""
        try:
            review_feedback = context.get_artifact("review_feedback")
            if not review_feedback or not isinstance(review_feedback, dict):
                console.print("[yellow]No review feedback available[/yellow]")
                return

            score = review_feedback.get("overall_score", "N/A")
            strengths = review_feedback.get("strengths", [])
            improvements = review_feedback.get("improvements", [])
            code_issues = review_feedback.get("code_issues", [])

            if strengths:
                strengths_text = "\n".join(f"  • {s}" for s in strengths[:3])
            else:
                strengths_text = "  (なし)"

            # Combine improvements and code_issues for display
            all_feedback = improvements + code_issues
            if all_feedback:
                feedback_text = "\n".join(
                    f"  • [{fb.get('section_index', '?')}] "
                    f"{fb.get('suggestion', fb.get('issue', ''))[:60]}..."
                    for fb in all_feedback[:5]
                )
            else:
                feedback_text = "  (なし)"

            console.print(
                Panel(
                    f"[bold]Score:[/bold] {score}/10\n\n"
                    f"[bold]Strengths:[/bold]\n{strengths_text}\n\n"
                    f"[bold]Feedback ({len(all_feedback)} items):[/bold]\n{feedback_text}",
                    title="[cyan]Review Output[/cyan]",
                    border_style="cyan",
                )
            )
        except Exception as e:
            console.print(f"[yellow]Could not display review output: {e}[/yellow]")

    def _build_article(self, context: StageContext) -> Article:
        """Build final article from stage outputs.

        Args:
            context: Pipeline context with artifacts

        Returns:
            Final article
        """
        # Get the final draft from context
        draft = context.get_artifact("draft")
        if isinstance(draft, Article):
            return draft

        # If draft stage returned something else, try to build from idea/outline
        idea: Optional[ArticleIdea] = context.get_artifact("ideation")
        # Note: outline is available via context.get_artifact("outline") if needed

        if idea is None:
            raise RuntimeError("No idea generated in pipeline")

        from ..models import ArticleFrontmatter

        frontmatter = ArticleFrontmatter(
            title=idea.title,
            emoji=idea.emoji,
            topics=idea.topics,
        )

        return Article(frontmatter=frontmatter)


def create_default_pipeline(
    llm_client: LLMClient,
    config: dict[str, Any],
    working_dir: Optional[Path] = None,
) -> ArticlePipeline:
    """Create a pipeline with default stages.

    Args:
        llm_client: LLM client for completions
        config: Pipeline configuration
        working_dir: Working directory for outputs

    Returns:
        Configured pipeline
    """
    from .draft import DraftStage
    from .ideation import IdeationStage
    from .outline import OutlineStage
    from .review import ReviewStage

    prompt_loader = PromptLoader(config.get("prompts", {}).get("dir", "config/prompts"))

    pipeline = ArticlePipeline(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        config=config,
        working_dir=working_dir,
    )

    pipeline.add_stage(IdeationStage())
    pipeline.add_stage(OutlineStage())
    pipeline.add_stage(DraftStage())
    pipeline.add_stage(ReviewStage())

    return pipeline


def create_project_pipeline(
    llm_client: LLMClient,
    config: dict[str, Any],
    working_dir: Optional[Path] = None,
) -> ArticlePipeline:
    """Create a pipeline with project analysis stage.

    This pipeline starts with project analysis using Claude Code SDK,
    then proceeds to ideation, outline, draft, and review stages.

    Args:
        llm_client: LLM client for completions
        config: Pipeline configuration
        working_dir: Working directory for outputs

    Returns:
        Configured pipeline with project analysis
    """
    from .draft import DraftStage
    from .ideation import IdeationStage
    from .outline import OutlineStage
    from .project_analysis import ProjectAnalysisStage
    from .review import ReviewStage

    prompt_loader = PromptLoader(config.get("prompts", {}).get("dir", "config/prompts"))

    pipeline = ArticlePipeline(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        config=config,
        working_dir=working_dir,
    )

    # プロジェクト分析を最初に実行
    pipeline.add_stage(ProjectAnalysisStage())
    pipeline.add_stage(IdeationStage())
    pipeline.add_stage(OutlineStage())
    pipeline.add_stage(DraftStage())
    pipeline.add_stage(ReviewStage())

    return pipeline
