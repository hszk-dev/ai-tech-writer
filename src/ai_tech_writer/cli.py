"""Command-line interface for AI Tech Writer."""

import asyncio
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

from .llm import LLMClient, PromptLoader
from .models import Platform
from .output import MarkdownRenderer, generate_filename
from .pipeline import ArticlePipeline
from .pipeline.draft import DraftStage
from .pipeline.ideation import IdeationStage
from .pipeline.outline import OutlineStage
from .pipeline.review import ReviewStage
from .treesearch import ArticleEvaluator, BFTSearch, SearchConfig

# Load environment variables
load_dotenv()

app = typer.Typer(
    name="ai-tech-writer",
    help="AI-powered technical article writer for Zenn/Qiita",
)
console = Console()


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file (defaults to config/default.yaml)

    Returns:
        Configuration dict
    """
    if config_path is None:
        config_path = Path("config/default.yaml")

    if not config_path.exists():
        console.print(f"[yellow]Config not found: {config_path}, using defaults[/yellow]")
        return {
            "llm": {
                "default_model": "claude-sonnet-4-20250514",
                "temperature": 0.7,
                "max_tokens": 4096,
            },
            "output": {
                "platform": "zenn",
                "output_dir": "outputs",
            },
            "web_search": {
                "provider": "tavily",
                "num_results": 10,
            },
            "prompts": {
                "dir": "config/prompts",
            },
        }

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.command()
def generate(
    topic: Optional[str] = typer.Argument(
        None, help="Topic for the article (optional if --project is used)"
    ),
    platform: str = typer.Option("zenn", "--platform", "-p", help="Target platform (zenn/qiita)"),
    output_dir: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="LLM model to use"),
    skip_review: bool = typer.Option(False, "--skip-review", help="Skip the review stage"),
    skip_search: bool = typer.Option(False, "--skip-search", help="Skip web search in ideation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed pipeline progress"),
    treesearch: bool = typer.Option(
        False, "--treesearch", "-t", help="Use Tree Search mode for better quality"
    ),
    beam_width: Optional[int] = typer.Option(
        None, "--beam-width", help="Tree Search beam width (default: 3)"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", help="Tree Search max iterations (default: 10)"
    ),
    max_depth: Optional[int] = typer.Option(
        None, "--max-depth", help="Tree Search max depth (default: 4)"
    ),
    project_path: Optional[str] = typer.Option(
        None, "--project", "-P", help="Local project path to analyze for article ideas"
    ),
    analysis_depth: str = typer.Option(
        "standard", "--depth", "-d", help="Project analysis depth: quick/standard/comprehensive"
    ),
):
    """Generate a technical article from a topic or local project."""
    # Validate inputs
    if not topic and not project_path:
        console.print("[red]Error: Either topic or --project must be specified[/red]")
        raise typer.Exit(1)

    display_topic = topic or f"Project: {project_path}"
    mode_info = ""
    if treesearch and project_path:
        mode_info = "\nMode: Tree Search + Project Analysis"
    elif treesearch:
        mode_info = "\nMode: Tree Search"
    elif project_path:
        mode_info = "\nMode: Project Analysis"

    console.print(
        Panel(
            f"[bold]AI Tech Writer[/bold]\n\nTopic: {display_topic}\nPlatform: {platform}"
            + (f"\nProject: {project_path}\nDepth: {analysis_depth}" if project_path else "")
            + mode_info,
            title="Starting",
        )
    )

    # Load config
    config = load_config(Path(config_path) if config_path else None)

    # Override with CLI options
    if output_dir:
        config["output"]["output_dir"] = output_dir
    if skip_search:
        config["web_search"]["provider"] = "none"
    if project_path:
        # Set analysis depth for project analysis
        if "project_analysis" not in config:
            config["project_analysis"] = {}
        config["project_analysis"]["analysis_depth"] = analysis_depth

    # Create LLM client
    llm_config = config.get("llm", {})
    llm_client = LLMClient(
        default_model=model or llm_config.get("default_model", "claude-sonnet-4-20250514"),
        temperature=llm_config.get("temperature", 0.7),
        max_tokens=llm_config.get("max_tokens", 4096),
    )

    # Create pipeline
    prompt_loader = PromptLoader(config.get("prompts", {}).get("dir", "config/prompts"))
    working_dir = Path(config.get("output", {}).get("output_dir", "outputs"))

    pipeline = ArticlePipeline(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        config=config,
        working_dir=working_dir,
        verbose=verbose,
    )

    # Add stages based on mode
    if project_path:
        # Use project analysis pipeline
        from .pipeline.project_analysis import ProjectAnalysisStage

        pipeline.add_stage(ProjectAnalysisStage())

    pipeline.add_stage(IdeationStage())
    pipeline.add_stage(OutlineStage())
    pipeline.add_stage(DraftStage())
    if not skip_review:
        pipeline.add_stage(ReviewStage())

    # Run pipeline
    try:
        if treesearch:
            # Use Tree Search mode (with optional project analysis)
            article = asyncio.run(
                _run_treesearch(
                    topic=topic or "",
                    llm_client=llm_client,
                    prompt_loader=prompt_loader,
                    config=config,
                    working_dir=working_dir,
                    verbose=verbose,
                    beam_width=beam_width,
                    max_iterations=max_iterations,
                    max_depth=max_depth,
                    project_path=project_path,
                )
            )
        elif project_path:
            # Use project analysis pipeline
            article = asyncio.run(
                pipeline.run_with_project(
                    project_path=project_path,
                    topic=topic,
                )
            )
        else:
            # Use standard pipeline
            article = asyncio.run(pipeline.run(topic or ""))

        # Render and save
        target_platform = Platform(platform)
        renderer = MarkdownRenderer()

        filename = generate_filename(article)
        output_path = working_dir / f"{filename}.md"

        renderer.save(article, output_path, target_platform)

        console.print(
            Panel(
                f"[green]Article generated successfully![/green]\n\n"
                f"Title: {article.frontmatter.title}\n"
                f"Output: {output_path}",
                title="Complete",
            )
        )

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


async def _run_treesearch(
    topic: str,
    llm_client: LLMClient,
    prompt_loader: PromptLoader,
    config: dict,
    working_dir: Path,
    verbose: bool,
    beam_width: Optional[int],
    max_iterations: Optional[int],
    max_depth: Optional[int],
    project_path: Optional[str] = None,
):
    """Run Tree Search mode for article generation.

    Args:
        topic: Article topic
        llm_client: LLM client
        prompt_loader: Prompt loader
        config: Configuration
        working_dir: Working directory
        verbose: Verbose output
        beam_width: Optional beam width override
        max_iterations: Optional max iterations override
        max_depth: Optional max depth override
        project_path: Optional project path to analyze before tree search

    Returns:
        Generated article
    """
    from datetime import datetime

    from .pipeline.base import StageContext

    console.print(
        Panel(
            "[bold cyan]Tree Search Mode[/bold cyan]\n\n"
            "Exploring multiple approaches to find the best article...",
            title="Mode",
        )
    )

    # Build search config
    treesearch_config = config.get("treesearch", {})
    search_config = SearchConfig.from_dict(treesearch_config)

    # Apply CLI overrides
    if beam_width is not None:
        search_config.beam_width = beam_width
    if max_iterations is not None:
        search_config.max_iterations = max_iterations
    if max_depth is not None:
        search_config.max_depth = max_depth

    # Create evaluator
    evaluator = ArticleEvaluator(llm_client, prompt_loader)

    # Create search instance
    search = BFTSearch(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        evaluator=evaluator,
        config=search_config,
        console=console,
        verbose=verbose,
    )

    # Create context
    context = StageContext(
        llm_client=llm_client,
        prompt_loader=prompt_loader,
        config=config,
        working_dir=working_dir,
    )

    # Run project analysis if project_path is provided
    if project_path:
        from .pipeline.project_analysis import ProjectAnalysisStage

        console.print(
            Panel(
                f"[bold cyan]Analyzing project before tree search...[/bold cyan]\n\n"
                f"Project: {project_path}",
                title="Project Analysis",
            )
        )

        project_analysis_stage = ProjectAnalysisStage()
        await project_analysis_stage.execute(
            {"project_path": project_path, "topic": topic},
            context,
        )
        console.print("[green]✓ Project analysis complete[/green]")

        # Derive topic from project analysis if not specified
        if not topic and context.project_analysis:
            if context.project_analysis.article_ideas:
                topic = context.project_analysis.article_ideas[0].title
            else:
                topic = context.project_analysis.project_name

    # Create intermediate output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_dir = working_dir / f".intermediate_{timestamp}"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Run search
    article = await search.search(topic, context, intermediate_dir)

    console.print(f"\n[dim]Intermediate files saved to: {intermediate_dir}[/dim]")

    return article


@app.command()
def list_models():
    """List available LLM models."""
    console.print("[bold]Available Models:[/bold]")
    console.print()
    console.print("[cyan]Anthropic (Claude):[/cyan]")
    console.print("  - claude-sonnet-4-20250514 (recommended)")
    console.print("  - claude-opus-4-20250514")
    console.print("  - claude-3-5-haiku-20241022")
    console.print()
    console.print("[cyan]OpenAI:[/cyan]")
    console.print("  - gpt-4o")
    console.print("  - gpt-4-turbo")
    console.print("  - gpt-4o-mini")


@app.command()
def init():
    """Initialize a new AI Tech Writer project."""
    console.print("[bold]Initializing AI Tech Writer project...[/bold]")

    # Create directories
    dirs = [
        "config/prompts",
        "templates/zenn",
        "templates/qiita",
        "outputs",
    ]

    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {d}/")

    # Create .env if not exists
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text(
            "# AI Tech Writer Environment\nANTHROPIC_API_KEY=sk-ant-xxx\nTAVILY_API_KEY=tvly-xxx\n"
        )
        console.print("  Created: .env")

    console.print()
    console.print("[green]Project initialized![/green]")
    console.print()
    console.print("Next steps:")
    console.print("  1. Add your API keys to .env")
    console.print("  2. Run: ai-tech-writer generate 'Your topic here'")


@app.command()
def analyze_project(
    project_path: str = typer.Argument(..., help="Path to the project directory"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output JSON file path"),
    config_path: Optional[str] = typer.Option(None, "--config", "-c", help="Path to config file"),
    depth: str = typer.Option(
        "standard", "--depth", "-d", help="Analysis depth: quick/standard/comprehensive"
    ),
):
    """Analyze a local project for article ideas.

    This command analyzes the given project using Claude Code SDK
    and extracts potential article ideas based on the project structure,
    code patterns, and notable implementations.
    """
    import json

    console.print(
        Panel(
            f"[bold]Project Analysis[/bold]\n\nProject: {project_path}\nDepth: {depth}",
            title="Starting",
        )
    )

    # Validate project path
    project_dir = Path(project_path).resolve()
    if not project_dir.exists():
        console.print(f"[red]Error: Project path does not exist: {project_dir}[/red]")
        raise typer.Exit(1)
    if not project_dir.is_dir():
        console.print(f"[red]Error: Project path is not a directory: {project_dir}[/red]")
        raise typer.Exit(1)

    # Load config
    config = load_config(Path(config_path) if config_path else None)

    # Update analysis depth
    if "project_analysis" not in config:
        config["project_analysis"] = {}
    config["project_analysis"]["analysis_depth"] = depth

    # Run analysis
    try:
        from .analysis import ClaudeCodeAnalyzer

        analyzer = ClaudeCodeAnalyzer(config.get("project_analysis", {}))
        analysis = asyncio.run(analyzer.analyze(project_dir))

        # Display results
        console.print(
            Panel(
                f"[bold]Project:[/bold] {analysis.project_name}\n"
                f"[bold]Description:[/bold] {analysis.description}\n"
                f"[bold]Tech Stack:[/bold] {', '.join(analysis.tech_stack.languages)}\n"
                f"[bold]Architecture:[/bold] {analysis.architecture.pattern}",
                title="[green]Analysis Complete[/green]",
            )
        )

        # Show article ideas
        console.print("\n[bold cyan]Article Ideas:[/bold cyan]")
        for i, idea in enumerate(analysis.article_ideas, 1):
            buzz_color = {
                "high": "green",
                "medium": "yellow",
                "low": "dim",
            }.get(idea.buzz_potential, "white")
            console.print(
                f"  {i}. [{buzz_color}][{idea.buzz_potential.upper()}][/{buzz_color}] {idea.title}"
            )
            console.print(f"     [dim]Target: {idea.target_audience}[/dim]")

        # Save to file if output specified
        if output:
            output_path = Path(output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)
            console.print(f"\n[dim]Analysis saved to: {output_path}[/dim]")

        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  Run: ai-tech-writer generate --project {project_path}")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
