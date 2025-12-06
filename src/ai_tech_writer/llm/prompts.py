"""Prompt template loader."""

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


class PromptLoader:
    """Load and render prompt templates."""

    def __init__(self, prompts_dir: str | Path = "config/prompts"):
        """Initialize prompt loader.

        Args:
            prompts_dir: Directory containing prompt templates (.md files)
        """
        self.prompts_dir = Path(prompts_dir)
        self._env: Optional[Environment] = None

    @property
    def env(self) -> Environment:
        """Get Jinja2 environment, creating if needed."""
        if self._env is None:
            self._env = Environment(
                loader=FileSystemLoader(str(self.prompts_dir)),
                autoescape=select_autoescape(enabled_extensions=()),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        return self._env

    def load(self, name: str, **variables: str) -> str:
        """Load and render a prompt template.

        Args:
            name: Template name (without .md extension)
            **variables: Variables to substitute in the template

        Returns:
            Rendered prompt string

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.prompts_dir / f"{name}.md"

        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        template = self.env.get_template(f"{name}.md")
        return template.render(**variables)

    def load_raw(self, name: str) -> str:
        """Load a prompt template without rendering.

        Args:
            name: Template name (without .md extension)

        Returns:
            Raw template content

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = self.prompts_dir / f"{name}.md"

        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        return template_path.read_text(encoding="utf-8")

    def list_templates(self) -> list[str]:
        """List available prompt templates.

        Returns:
            List of template names (without .md extension)
        """
        if not self.prompts_dir.exists():
            return []

        return [p.stem for p in self.prompts_dir.glob("*.md")]
