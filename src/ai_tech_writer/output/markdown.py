"""Markdown renderer using Jinja2 templates."""

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..models import Article, Platform


class MarkdownRenderer:
    """Render articles to Markdown using templates."""

    def __init__(self, templates_dir: str | Path = "templates"):
        """Initialize renderer.

        Args:
            templates_dir: Directory containing platform-specific templates
        """
        self.templates_dir = Path(templates_dir)
        self._env: Optional[Environment] = None

    @property
    def env(self) -> Environment:
        """Get Jinja2 environment."""
        if self._env is None:
            self._env = Environment(
                loader=FileSystemLoader(str(self.templates_dir)),
                autoescape=select_autoescape(enabled_extensions=()),
                trim_blocks=True,
                lstrip_blocks=True,
            )
            # Add custom filters
            self._env.filters["lower"] = str.lower
        return self._env

    def render(
        self,
        article: Article,
        platform: Platform = Platform.ZENN,
    ) -> str:
        """Render article to Markdown.

        Args:
            article: Article to render
            platform: Target platform (zenn or qiita)

        Returns:
            Rendered Markdown string
        """
        template_path = f"{platform.value}/article.md.j2"

        try:
            template = self.env.get_template(template_path)
            return template.render(
                frontmatter=article.frontmatter,
                sections=article.sections,
                references=article.references,
            )
        except Exception:
            # Fallback to built-in rendering
            return article.to_markdown(platform)

    def save(
        self,
        article: Article,
        output_path: str | Path,
        platform: Platform = Platform.ZENN,
    ) -> Path:
        """Render and save article to file.

        Args:
            article: Article to save
            output_path: Output file path
            platform: Target platform

        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.render(article, platform)
        output_path.write_text(content, encoding="utf-8")

        return output_path


def generate_filename(article: Article) -> str:
    """Generate a filename from article title.

    Args:
        article: Article to generate filename for

    Returns:
        Safe filename (without extension)
    """
    import re
    from datetime import datetime

    # Create slug from title
    title = article.frontmatter.title
    # Remove special characters
    slug = re.sub(r"[^\w\s-]", "", title)
    # Replace spaces with hyphens
    slug = re.sub(r"[\s_]+", "-", slug)
    # Lowercase
    slug = slug.lower().strip("-")
    # Limit length
    slug = slug[:50]

    # Add date prefix
    date_str = datetime.now().strftime("%Y%m%d")

    return f"{date_str}-{slug}"
