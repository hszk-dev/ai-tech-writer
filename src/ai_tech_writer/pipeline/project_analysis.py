"""Project analysis pipeline stage."""

from pathlib import Path
from typing import Any

from rich.console import Console

from ..analysis import ClaudeCodeAnalyzer, ProjectAnalysis
from .base import PipelineStage, StageContext

console = Console()


class ProjectAnalysisStage(PipelineStage[dict[str, Any], ProjectAnalysis]):
    """プロジェクト分析ステージ.

    ローカルプロジェクトをClaude Code SDKで分析し、
    技術スタック、アーキテクチャ、記事ネタを抽出する。
    """

    name = "project_analysis"
    description = "Analyzing local project structure and code"

    async def execute(
        self,
        input_data: dict[str, Any],
        context: StageContext,
    ) -> ProjectAnalysis:
        """プロジェクトを分析.

        Args:
            input_data: {"project_path": "/path/to/project", "topic": "optional topic"}
            context: パイプラインコンテキスト

        Returns:
            ProjectAnalysis: 分析結果
        """
        project_path = Path(input_data.get("project_path", ".")).resolve()

        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        if not project_path.is_dir():
            raise ValueError(f"Project path is not a directory: {project_path}")

        console.print(f"[bold blue]Analyzing project:[/bold blue] {project_path}")

        # Claude Code Analyzerで分析
        analyzer_config = context.config.get("project_analysis", {})
        analyzer = ClaudeCodeAnalyzer(analyzer_config)

        analysis = await analyzer.analyze(project_path)

        # コンテキストに保存
        context.project_analysis = analysis
        context.project_path = project_path
        context.save_artifact("project_analysis", analysis.to_dict())

        # サマリーを表示
        console.print("\n[bold green]Analysis Complete![/bold green]")
        console.print(f"  Project: {analysis.project_name}")
        console.print(f"  Tech Stack: {', '.join(analysis.tech_stack.languages)}")
        console.print(f"  Article Ideas: {len(analysis.article_ideas)} found")

        return analysis

    def validate_input(self, input_data: dict[str, Any]) -> bool:
        """入力検証."""
        if not isinstance(input_data, dict):
            return False
        project_path = input_data.get("project_path")
        if not project_path:
            return False
        return True
