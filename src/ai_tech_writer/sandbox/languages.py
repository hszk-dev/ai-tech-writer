"""Language configurations for code validation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LanguageConfig:
    """Configuration for a programming language."""

    name: str
    extensions: list[str]
    syntax_checker: Optional[str] = None  # "ast" for Python, "esprima" for JS, etc.
    docker_image: Optional[str] = None
    run_command: Optional[str] = None
    aliases: list[str] | None = None

    def matches(self, lang: str) -> bool:
        """Check if this config matches a language identifier."""
        lang_lower = lang.lower()
        if lang_lower == self.name.lower():
            return True
        if self.aliases:
            return lang_lower in [a.lower() for a in self.aliases]
        return False


# Supported language configurations
LANGUAGE_CONFIGS = [
    LanguageConfig(
        name="python",
        extensions=[".py"],
        syntax_checker="ast",
        docker_image="python:3.11-slim",
        run_command="python {file}",
        aliases=["py", "python3"],
    ),
    LanguageConfig(
        name="javascript",
        extensions=[".js"],
        syntax_checker="acorn",
        docker_image="node:20-slim",
        run_command="node {file}",
        aliases=["js", "node"],
    ),
    LanguageConfig(
        name="typescript",
        extensions=[".ts"],
        syntax_checker="typescript",
        docker_image="node:20-slim",
        run_command="npx ts-node {file}",
        aliases=["ts"],
    ),
    LanguageConfig(
        name="bash",
        extensions=[".sh", ".bash"],
        syntax_checker="bash",
        docker_image="bash:5",
        run_command="bash {file}",
        aliases=["sh", "shell"],
    ),
    LanguageConfig(
        name="json",
        extensions=[".json"],
        syntax_checker="json",
        aliases=["jsonc"],
    ),
    LanguageConfig(
        name="yaml",
        extensions=[".yaml", ".yml"],
        syntax_checker="yaml",
        aliases=["yml"],
    ),
    LanguageConfig(
        name="html",
        extensions=[".html", ".htm"],
        syntax_checker=None,  # HTML is very lenient
    ),
    LanguageConfig(
        name="css",
        extensions=[".css"],
        syntax_checker=None,
    ),
    LanguageConfig(
        name="sql",
        extensions=[".sql"],
        syntax_checker=None,
    ),
    LanguageConfig(
        name="rust",
        extensions=[".rs"],
        syntax_checker=None,
        docker_image="rust:slim",
        run_command="rustc {file} -o /tmp/out && /tmp/out",
    ),
    LanguageConfig(
        name="go",
        extensions=[".go"],
        syntax_checker=None,
        docker_image="golang:1.21-alpine",
        run_command="go run {file}",
    ),
]


# Quick lookup set
SUPPORTED_LANGUAGES = {cfg.name for cfg in LANGUAGE_CONFIGS}
for cfg in LANGUAGE_CONFIGS:
    if cfg.aliases:
        SUPPORTED_LANGUAGES.update(cfg.aliases)


def get_language_config(language: str) -> Optional[LanguageConfig]:
    """Get configuration for a language.

    Args:
        language: Language name or alias

    Returns:
        LanguageConfig if found, None otherwise
    """
    for config in LANGUAGE_CONFIGS:
        if config.matches(language):
            return config
    return None
