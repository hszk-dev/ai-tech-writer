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
    topic: str = typer.Argument(..., help="Topic for the article"),
    platform: str = typer.Option(
        "zenn", "--platform", "-p", help="Target platform (zenn/qiita)"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    config_path: Optional[str] = typer.Option(
        None, "--config", "-c", help="Path to config file"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model to use"
    ),
    skip_review: bool = typer.Option(
        False, "--skip-review", help="Skip the review stage"
    ),
    skip_search: bool = typer.Option(
        False, "--skip-search", help="Skip web search in ideation"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed pipeline progress"
    ),
):
    """Generate a technical article from a topic."""
    console.print(
        Panel(
            f"[bold]AI Tech Writer[/bold]\n\nTopic: {topic}\nPlatform: {platform}",
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

    # Add stages
    pipeline.add_stage(IdeationStage())
    pipeline.add_stage(OutlineStage())
    pipeline.add_stage(DraftStage())
    if not skip_review:
        pipeline.add_stage(ReviewStage())

    # Run pipeline
    try:
        article = asyncio.run(pipeline.run(topic))

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
            "# AI Tech Writer Environment\n"
            "ANTHROPIC_API_KEY=sk-ant-xxx\n"
            "TAVILY_API_KEY=tvly-xxx\n"
        )
        console.print("  Created: .env")

    console.print()
    console.print("[green]Project initialized![/green]")
    console.print()
    console.print("Next steps:")
    console.print("  1. Add your API keys to .env")
    console.print("  2. Run: ai-tech-writer generate 'Your topic here'")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
