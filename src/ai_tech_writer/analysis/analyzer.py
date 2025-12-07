"""Claude Code SDK wrapper for project analysis."""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from rich.console import Console

from .models import (
    ArchitectureInfo,
    ArticleIdeaCandidate,
    NotableImplementation,
    ProjectAnalysis,
    TechStackInfo,
)

logger = logging.getLogger(__name__)
console = Console()

# depth設定マッピング
DEPTH_CONFIG = {
    "quick": {"max_turns": 10, "tree_depth": 2, "min_ideas": 2},
    "standard": {"max_turns": 20, "tree_depth": 3, "min_ideas": 3},
    "comprehensive": {"max_turns": 30, "tree_depth": 4, "min_ideas": 5},
}


class ClaudeCodeAnalyzer:
    """Claude Code SDKを使ったプロジェクト分析."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_files = config.get("max_files", 100)
        self.include_patterns = config.get(
            "include_patterns", ["*.py", "*.ts", "*.js", "*.go", "*.rs"]
        )
        self.exclude_patterns = config.get(
            "exclude_patterns",
            ["node_modules/**", ".git/**", "__pycache__/**", "*.lock"],
        )
        self.analysis_depth = config.get("analysis_depth", "standard")

    async def analyze(self, project_path: Path) -> ProjectAnalysis:
        """プロジェクトを分析."""
        console.print(f"[blue]Analyzing project:[/blue] {project_path}")

        # 1. ファイルツリーの取得
        file_tree = self._get_file_tree(project_path)

        # 2. READMEの読み取り
        readme_content = self._read_readme(project_path)

        # 3. Claude Code SDKで分析
        analysis_result = await self._analyze_with_sdk(project_path, file_tree, readme_content)

        # 4. ProjectAnalysisオブジェクトの構築
        return self._build_analysis(project_path, file_tree, readme_content, analysis_result)

    def _get_file_tree(self, project_path: Path) -> str:
        """ファイルツリーを取得."""
        depth_config = DEPTH_CONFIG.get(self.analysis_depth, DEPTH_CONFIG["standard"])
        tree_depth = depth_config["tree_depth"]

        try:
            result = subprocess.run(
                ["tree", "-L", str(tree_depth), "-I", "node_modules|.git|__pycache__|.venv"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout[:5000]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # 簡易ファイルリスト
        lines = []
        count = 0
        for pattern in self.include_patterns:
            for file in project_path.glob(f"**/{pattern}"):
                if count >= self.max_files:
                    lines.append(f"... and more ({count}+ files)")
                    break
                rel_path = file.relative_to(project_path)
                if any(rel_path.match(exc) for exc in self.exclude_patterns):
                    continue
                lines.append(str(rel_path))
                count += 1
        return "\n".join(lines)

    def _read_readme(self, project_path: Path) -> str:
        """READMEファイルを読み取り."""
        for name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = project_path / name
            if readme_path.exists():
                try:
                    return readme_path.read_text(encoding="utf-8")[:10000]
                except Exception as e:
                    logger.debug(f"Failed to read README at {readme_path}: {e}")
        return ""

    async def _analyze_with_sdk(
        self,
        project_path: Path,
        file_tree: str,
        readme_content: str,
    ) -> dict[str, Any]:
        """Claude Code SDKで分析."""
        from claude_code_sdk import ClaudeCodeOptions, query

        depth_config = DEPTH_CONFIG.get(self.analysis_depth, DEPTH_CONFIG["standard"])
        prompt = self._build_prompt(project_path, file_tree, readme_content)

        options = ClaudeCodeOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            max_turns=depth_config["max_turns"],
            cwd=str(project_path),
        )

        console.print(f"[dim]Running Claude Code analysis (depth: {self.analysis_depth})...[/dim]")

        result_text = ""
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        result_text += block.text

        if not result_text.strip():
            raise RuntimeError("Claude Code SDK returned empty result")

        parsed = self._parse_json_result(result_text)
        if not parsed:
            # JSONが見つからない場合、より詳細なエラーメッセージを表示
            console.print(f"[yellow]Response text ({len(result_text)} chars):[/yellow]")
            preview = result_text[:1000] + "..." if len(result_text) > 1000 else result_text
            console.print(f"[dim]{preview}[/dim]")
            raise RuntimeError(
                "Claude Code SDKがJSON形式で応答しませんでした。"
                "max_turnsが不足している可能性があります。"
            )

        return parsed

    def _build_prompt(
        self,
        project_path: Path,
        file_tree: str,
        readme_content: str,
    ) -> str:
        """分析用プロンプトを構築."""
        depth_config = DEPTH_CONFIG.get(self.analysis_depth, DEPTH_CONFIG["standard"])
        min_ideas = depth_config["min_ideas"]

        # depth別のタスク説明
        if self.analysis_depth == "quick":
            task = f"""## 分析タスク（簡易モード）
1. 技術スタックを特定
2. バズりそうな記事アイデアを{min_ideas}つ提案"""
        elif self.analysis_depth == "comprehensive":
            task = f"""## 分析タスク（詳細モード）
1. 主要ファイルを詳細に読んで技術スタックを特定
2. アーキテクチャパターンを詳細に分析
3. 注目すべき実装を発見し、コードスニペットを抽出
4. バズりそうな記事アイデアを{min_ideas}つ以上提案（各アイデアに具体的なコード参照を含める）"""
        else:  # standard
            task = f"""## 分析タスク
1. 主要ファイルを読んで技術スタックを特定
2. アーキテクチャパターンを分析
3. 注目すべき実装を発見
4. バズりそうな記事アイデアを{min_ideas}つ以上提案"""

        return f"""以下のプロジェクトを分析し、Zenn/Qiitaでバズりそうな記事ネタを見つけてください。

## プロジェクトパス
{project_path}

## ファイル構造
```
{file_tree[:3000]}
```

## README
```markdown
{readme_content[:5000]}
```

{task}

## 回答形式
必ず以下のJSON形式で回答してください：

```json
{{
  "project_name": "プロジェクト名",
  "description": "プロジェクトの説明",
  "tech_stack": {{
    "languages": ["Python"],
    "frameworks": ["FastAPI"],
    "libraries": ["Pydantic", "Rich"],
    "tools": ["Docker"]
  }},
  "architecture": {{
    "pattern": "パイプラインアーキテクチャ",
    "components": ["Pipeline", "Stages"],
    "layers": ["Application", "Domain"],
    "design_decisions": ["非同期処理", "ステージ分離"]
  }},
  "notable_implementations": [
    {{
      "feature": "機能名",
      "description": "優れている点",
      "files": ["src/file.py"]
    }}
  ],
  "article_ideas": [
    {{
      "title": "バズりそうなタイトル",
      "angle": "記事の切り口",
      "target_audience": "中級者向け",
      "buzz_potential": "high",
      "novelty_points": ["新規性1", "新規性2"],
      "key_points": ["ポイント1"],
      "recommended_snippets": ["src/file.py"]
    }}
  ]
}}
```

重要:
- バズりそうな記事は「新規性」と「実用性」を重視
- 2025年のトレンドを意識
- タイトルに具体的な数字や事例を含める

【最重要】分析完了後、必ず上記のJSON形式で結果を出力してください。JSONブロック以外の説明は不要です。
"""

    def _parse_json_result(self, result_text: str) -> dict[str, Any]:
        """JSON結果をパース."""
        # ```json ブロックを抽出
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", result_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 直接パース
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            pass

        # { } を探す
        brace_match = re.search(r"\{[\s\S]*\}", result_text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return {}

    def _build_analysis(
        self,
        project_path: Path,
        file_tree: str,
        readme_content: str,
        result: dict[str, Any],
    ) -> ProjectAnalysis:
        """分析結果からProjectAnalysisオブジェクトを構築."""
        tech_stack_data = result.get("tech_stack", {})
        architecture_data = result.get("architecture", {})

        notable_implementations = [
            NotableImplementation(
                feature=impl.get("feature", ""),
                description=impl.get("description", ""),
                files=impl.get("files", []),
                code_snippets=[],
            )
            for impl in result.get("notable_implementations", [])
        ]

        article_ideas = [
            ArticleIdeaCandidate(
                title=idea.get("title", ""),
                angle=idea.get("angle", ""),
                target_audience=idea.get("target_audience", ""),
                buzz_potential=idea.get("buzz_potential", "medium"),
                novelty_points=idea.get("novelty_points", []),
                key_points=idea.get("key_points", []),
                recommended_snippets=idea.get("recommended_snippets", []),
            )
            for idea in result.get("article_ideas", [])
        ]

        return ProjectAnalysis(
            project_path=project_path,
            project_name=result.get("project_name", project_path.name),
            description=result.get("description", ""),
            tech_stack=TechStackInfo.from_dict(tech_stack_data),
            architecture=ArchitectureInfo.from_dict(architecture_data),
            notable_implementations=notable_implementations,
            article_ideas=article_ideas,
            file_tree=file_tree,
            readme_content=readme_content,
        )
