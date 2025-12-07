"""Article domain models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ArticleType(str, Enum):
    """Article type for Zenn."""

    TECH = "tech"
    IDEA = "idea"


class Platform(str, Enum):
    """Target platform."""

    ZENN = "zenn"
    QIITA = "qiita"


@dataclass
class ArticleFrontmatter:
    """Frontmatter for Zenn/Qiita articles."""

    title: str
    emoji: str = "📝"
    type: ArticleType = ArticleType.TECH
    topics: list[str] = field(default_factory=list)
    published: bool = False

    def to_zenn_frontmatter(self) -> str:
        """Convert to Zenn frontmatter format."""
        topics_str = ", ".join(f'"{t}"' for t in self.topics[:5])
        return f'''---
title: "{self.title}"
emoji: "{self.emoji}"
type: "{self.type.value}"
topics: [{topics_str}]
published: {str(self.published).lower()}
---'''

    def to_qiita_frontmatter(self) -> str:
        """Convert to Qiita frontmatter format."""
        tags = "\n".join(f"  - {t}" for t in self.topics[:5])
        return f"""---
title: {self.title}
tags:
{tags}
private: {str(not self.published).lower()}
---"""


@dataclass
class CodeExample:
    """Code example within an article section."""

    language: str
    code: str
    description: str = ""
    filename: str = ""
    output: Optional[str] = None

    def to_markdown(self) -> str:
        """Convert to markdown code block."""
        filename_suffix = f":{self.filename}" if self.filename else ""
        result = f"```{self.language}{filename_suffix}\n{self.code}\n```"
        if self.output:
            result += f"\n\n実行結果:\n```\n{self.output}\n```"
        return result


@dataclass
class ArticleSection:
    """A section of the article."""

    heading: str
    level: int  # 1-6 for h1-h6
    content: str
    code_examples: list[CodeExample] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Convert to markdown."""
        heading_md = "#" * self.level + " " + self.heading
        parts = [heading_md, "", self.content]

        for code in self.code_examples:
            parts.append("")
            parts.append(code.to_markdown())

        return "\n".join(parts)


@dataclass
class Article:
    """Complete article representation."""

    frontmatter: ArticleFrontmatter
    sections: list[ArticleSection] = field(default_factory=list)
    references: list[str] = field(default_factory=list)

    def to_markdown(self, platform: Platform = Platform.ZENN) -> str:
        """Render to platform-specific markdown."""
        parts = []

        # Frontmatter
        if platform == Platform.ZENN:
            parts.append(self.frontmatter.to_zenn_frontmatter())
        else:
            parts.append(self.frontmatter.to_qiita_frontmatter())

        parts.append("")

        # Sections
        for section in self.sections:
            parts.append(section.to_markdown())
            parts.append("")

        # References
        if self.references:
            parts.append("## 参考文献")
            parts.append("")
            for ref in self.references:
                parts.append(f"- {ref}")

        return "\n".join(parts)


# --- Intermediate data structures for pipeline ---


@dataclass
class OutlineSection:
    """Section in an article outline."""

    heading: str
    key_points: list[str] = field(default_factory=list)
    code_needed: bool = False
    estimated_words: int = 200


@dataclass
class ArticleIdea:
    """Generated article idea from Ideation stage."""

    title: str
    emoji: str
    topics: list[str]
    target_audience: str
    problem_to_solve: str
    key_takeaways: list[str]
    suggested_sections: list[str]
    # バズ・新規性関連（オプション）
    novelty_points: list[str] = field(default_factory=list)
    buzz_factors: list[str] = field(default_factory=list)
    trend_relevance: str = "medium"  # high/medium/low
    hook_elements: list[str] = field(default_factory=list)
    # プロジェクトベース記事用（オプション）
    code_references: list[dict[str, str]] = field(default_factory=list)
    project_context: Optional[str] = None


@dataclass
class ArticleOutline:
    """Detailed outline from Outline stage."""

    idea: ArticleIdea
    sections: list[OutlineSection]
    introduction: str
    conclusion_points: list[str]
