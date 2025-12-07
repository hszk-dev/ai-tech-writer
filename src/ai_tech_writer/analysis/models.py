"""Data models for project analysis."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TechStackInfo:
    """技術スタック情報."""

    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "languages": self.languages,
            "frameworks": self.frameworks,
            "libraries": self.libraries,
            "tools": self.tools,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TechStackInfo":
        return cls(
            languages=data.get("languages", []),
            frameworks=data.get("frameworks", []),
            libraries=data.get("libraries", []),
            tools=data.get("tools", []),
        )


@dataclass
class ArchitectureInfo:
    """アーキテクチャ情報."""

    pattern: str = ""
    components: list[str] = field(default_factory=list)
    layers: list[str] = field(default_factory=list)
    design_decisions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": self.pattern,
            "components": self.components,
            "layers": self.layers,
            "design_decisions": self.design_decisions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitectureInfo":
        return cls(
            pattern=data.get("pattern", ""),
            components=data.get("components", []),
            layers=data.get("layers", []),
            design_decisions=data.get("design_decisions", []),
        )


@dataclass
class CodeSnippet:
    """コードスニペット."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content": self.content,
            "language": self.language,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeSnippet":
        return cls(
            file_path=data.get("file_path", ""),
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            content=data.get("content", ""),
            language=data.get("language", ""),
            description=data.get("description", ""),
        )


@dataclass
class NotableImplementation:
    """注目すべき実装."""

    feature: str
    description: str
    files: list[str] = field(default_factory=list)
    code_snippets: list[CodeSnippet] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "description": self.description,
            "files": self.files,
            "code_snippets": [s.to_dict() for s in self.code_snippets],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NotableImplementation":
        return cls(
            feature=data.get("feature", ""),
            description=data.get("description", ""),
            files=data.get("files", []),
            code_snippets=[CodeSnippet.from_dict(s) for s in data.get("code_snippets", [])],
        )


@dataclass
class ArticleIdeaCandidate:
    """記事アイデア候補."""

    title: str
    angle: str
    target_audience: str
    buzz_potential: str  # "high", "medium", "low"
    novelty_points: list[str] = field(default_factory=list)
    key_points: list[str] = field(default_factory=list)
    recommended_snippets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "angle": self.angle,
            "target_audience": self.target_audience,
            "buzz_potential": self.buzz_potential,
            "novelty_points": self.novelty_points,
            "key_points": self.key_points,
            "recommended_snippets": self.recommended_snippets,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArticleIdeaCandidate":
        return cls(
            title=data.get("title", ""),
            angle=data.get("angle", ""),
            target_audience=data.get("target_audience", ""),
            buzz_potential=data.get("buzz_potential", "medium"),
            novelty_points=data.get("novelty_points", []),
            key_points=data.get("key_points", []),
            recommended_snippets=data.get("recommended_snippets", []),
        )


@dataclass
class ProjectAnalysis:
    """プロジェクト分析結果."""

    project_path: Path
    project_name: str
    description: str
    tech_stack: TechStackInfo
    architecture: ArchitectureInfo
    notable_implementations: list[NotableImplementation]
    article_ideas: list[ArticleIdeaCandidate]
    file_tree: str = ""
    readme_content: str = ""
    analyzed_at: str = ""

    def __post_init__(self):
        if not self.analyzed_at:
            self.analyzed_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_path": str(self.project_path),
            "project_name": self.project_name,
            "description": self.description,
            "tech_stack": self.tech_stack.to_dict(),
            "architecture": self.architecture.to_dict(),
            "notable_implementations": [impl.to_dict() for impl in self.notable_implementations],
            "article_ideas": [idea.to_dict() for idea in self.article_ideas],
            "file_tree": self.file_tree,
            "readme_content": self.readme_content,
            "analyzed_at": self.analyzed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectAnalysis":
        return cls(
            project_path=Path(data.get("project_path", ".")),
            project_name=data.get("project_name", ""),
            description=data.get("description", ""),
            tech_stack=TechStackInfo.from_dict(data.get("tech_stack", {})),
            architecture=ArchitectureInfo.from_dict(data.get("architecture", {})),
            notable_implementations=[
                NotableImplementation.from_dict(impl)
                for impl in data.get("notable_implementations", [])
            ],
            article_ideas=[
                ArticleIdeaCandidate.from_dict(idea) for idea in data.get("article_ideas", [])
            ],
            file_tree=data.get("file_tree", ""),
            readme_content=data.get("readme_content", ""),
            analyzed_at=data.get("analyzed_at", ""),
        )

    def get_summary(self) -> str:
        """分析結果のサマリーを取得."""
        lines = [
            f"# {self.project_name}",
            "",
            f"**説明**: {self.description}",
            "",
            "## 技術スタック",
            f"- 言語: {', '.join(self.tech_stack.languages)}",
            f"- フレームワーク: {', '.join(self.tech_stack.frameworks)}",
            f"- ライブラリ: {', '.join(self.tech_stack.libraries)}",
            "",
            "## アーキテクチャ",
            f"- パターン: {self.architecture.pattern}",
            f"- コンポーネント: {', '.join(self.architecture.components)}",
            "",
            "## 注目ポイント",
        ]
        for impl in self.notable_implementations[:5]:
            lines.append(f"- **{impl.feature}**: {impl.description}")

        lines.extend(["", "## 記事アイデア候補"])
        for idea in self.article_ideas[:3]:
            lines.append(f"- [{idea.buzz_potential.upper()}] {idea.title}")

        return "\n".join(lines)
