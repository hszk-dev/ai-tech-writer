"""Review stage - reviews and improves the article draft."""

import re
from dataclasses import dataclass, asdict
from typing import Optional

from ..llm import Message
from ..models import Article, ArticleSection, CodeExample
from ..sandbox import (
    CodeValidator,
    ValidationResult,
    validate_code_blocks,
    DockerSandbox,
    ExecutionResult,
    get_language_config,
)
from .base import PipelineStage, StageContext


@dataclass
class ReviewResult:
    """Result of a review iteration."""

    article: Article
    score: float
    needs_improvement: bool
    feedback: dict
    iteration: int


@dataclass
class CodeValidationSummary:
    """Summary of code validation results."""

    total_blocks: int
    valid_blocks: int
    invalid_blocks: int
    errors: list[dict]


@dataclass
class CodeExecutionSummary:
    """Summary of code execution results."""

    total_executed: int
    successful: int
    failed: int
    skipped: int
    results: list[dict]


class ReviewStage(PipelineStage[Article, Article]):
    """Stage for reviewing and improving article draft."""

    name = "review"
    description = "Reviewing and improving article"

    async def execute(
        self,
        input_data: Article,
        context: StageContext,
    ) -> Article:
        """Review and improve article with iterative refinement.

        Args:
            input_data: Article draft
            context: Pipeline context

        Returns:
            Improved article
        """
        # Get iteration settings from config
        pipeline_config = context.config.get("pipeline", {})
        max_revisions = pipeline_config.get("max_revisions", 3)
        score_threshold = pipeline_config.get("revision_threshold", 8.0)
        validate_code = pipeline_config.get("validate_code", True)
        execute_code = pipeline_config.get("execute_code", False)

        current_article = input_data
        all_feedback: list[dict] = []

        # Validate code blocks before review
        code_validation = None
        if validate_code:
            code_validation = self._validate_code_blocks(current_article)
            context.save_artifact("code_validation", asdict(code_validation))

        # Execute code blocks if enabled
        code_execution = None
        if execute_code:
            code_execution = self._execute_code_blocks(current_article, context)
            context.save_artifact("code_execution", asdict(code_execution))

        for iteration in range(max_revisions):
            # Generate review feedback (including code validation/execution results)
            review_result = await self._generate_review(
                current_article, context, code_validation, code_execution
            )
            all_feedback.append(review_result)

            score = float(review_result.get("overall_score", 0))
            needs_improvement = review_result.get("needs_improvement", False)

            # Save current iteration feedback
            context.save_artifact(f"review_feedback_{iteration}", review_result)

            # Check if we've reached the quality threshold
            if score >= score_threshold or not needs_improvement:
                context.save_artifact("review_feedback", review_result)
                context.save_artifact("review_iterations", iteration + 1)
                context.save_artifact("final_score", score)
                return current_article

            # Apply improvements
            current_article = await self._apply_improvements(
                current_article, review_result, context
            )

            # Re-validate and re-execute code after improvements
            if validate_code:
                code_validation = self._validate_code_blocks(current_article)
            if execute_code:
                code_execution = self._execute_code_blocks(current_article, context)

        # Save final feedback after all iterations
        context.save_artifact("review_feedback", all_feedback[-1] if all_feedback else {})
        context.save_artifact("review_iterations", max_revisions)
        context.save_artifact("final_score", float(all_feedback[-1].get("overall_score", 0)) if all_feedback else 0)

        return current_article

    def _validate_code_blocks(self, article: Article) -> CodeValidationSummary:
        """Validate all code blocks in the article.

        Args:
            article: Article to validate

        Returns:
            CodeValidationSummary with validation results
        """
        markdown = article.to_markdown()
        results = validate_code_blocks(markdown)

        errors = []
        valid_count = 0
        invalid_count = 0

        for result in results:
            if result.valid:
                valid_count += 1
            else:
                invalid_count += 1
                errors.append({
                    "language": result.language,
                    "error": result.error_message,
                    "line": result.line_number,
                    "suggestions": result.suggestions or [],
                })

        return CodeValidationSummary(
            total_blocks=len(results),
            valid_blocks=valid_count,
            invalid_blocks=invalid_count,
            errors=errors,
        )

    def _execute_code_blocks(
        self,
        article: Article,
        context: StageContext,
    ) -> CodeExecutionSummary:
        """Execute code blocks in Docker sandbox.

        Args:
            article: Article with code blocks
            context: Pipeline context

        Returns:
            CodeExecutionSummary with execution results
        """
        # Get sandbox config
        sandbox_config = context.config.get("sandbox", {})
        timeout = sandbox_config.get("timeout", 30)
        memory_limit = sandbox_config.get("memory_limit", "256m")
        cpu_limit = sandbox_config.get("cpu_limit", 0.5)
        allowed_languages = sandbox_config.get("languages", [])

        # Initialize Docker sandbox
        sandbox = DockerSandbox(
            timeout=timeout,
            memory_limit=memory_limit,
            cpu_limit=cpu_limit,
        )

        # Check if Docker is available
        if not sandbox.is_available():
            return CodeExecutionSummary(
                total_executed=0,
                successful=0,
                failed=0,
                skipped=0,
                results=[{"error": "Docker is not available"}],
            )

        # Extract code blocks from markdown
        markdown = article.to_markdown()
        code_block_pattern = r"```(\w+)?\n(.*?)```"
        code_blocks = re.findall(code_block_pattern, markdown, re.DOTALL)

        results = []
        successful = 0
        failed = 0
        skipped = 0

        for i, (language, code) in enumerate(code_blocks):
            if not language:
                skipped += 1
                continue

            # Check if language is in allowed list (if specified)
            if allowed_languages and language.lower() not in [l.lower() for l in allowed_languages]:
                skipped += 1
                continue

            # Check if language has Docker support
            lang_config = get_language_config(language)
            if not lang_config or not lang_config.docker_image:
                skipped += 1
                continue

            # Execute code
            exec_result = sandbox.execute(code.strip(), language)

            result_dict = {
                "block_index": i,
                "language": language,
                "success": exec_result.success,
                "output": exec_result.output[:500] if exec_result.output else "",
                "error": exec_result.error[:500] if exec_result.error else "",
                "timed_out": exec_result.timed_out,
            }
            results.append(result_dict)

            if exec_result.success:
                successful += 1
            else:
                failed += 1

        return CodeExecutionSummary(
            total_executed=successful + failed,
            successful=successful,
            failed=failed,
            skipped=skipped,
            results=results,
        )

    async def _generate_review(
        self,
        article: Article,
        context: StageContext,
        code_validation: Optional[CodeValidationSummary] = None,
        code_execution: Optional[CodeExecutionSummary] = None,
    ) -> dict:
        """Generate review feedback for the article.

        Args:
            article: Article to review
            context: Pipeline context
            code_validation: Optional code validation results
            code_execution: Optional code execution results

        Returns:
            Review feedback dict
        """
        # Convert article to markdown for review
        article_content = article.to_markdown()

        # Add code validation info to prompt if available
        code_validation_text = ""
        if code_validation and code_validation.invalid_blocks > 0:
            code_validation_text = "\n\n## コード検証結果（構文チェック）\n"
            code_validation_text += f"検証済みコードブロック: {code_validation.total_blocks}\n"
            code_validation_text += f"有効: {code_validation.valid_blocks}, 無効: {code_validation.invalid_blocks}\n"
            if code_validation.errors:
                code_validation_text += "\n### 構文エラー:\n"
                for err in code_validation.errors:
                    code_validation_text += f"- [{err['language']}] {err['error']}"
                    if err.get('line'):
                        code_validation_text += f" (行: {err['line']})"
                    code_validation_text += "\n"

        # Add code execution info to prompt if available
        code_execution_text = ""
        if code_execution and code_execution.total_executed > 0:
            code_execution_text = "\n\n## コード実行結果（Docker sandbox）\n"
            code_execution_text += f"実行済み: {code_execution.total_executed}, "
            code_execution_text += f"成功: {code_execution.successful}, 失敗: {code_execution.failed}, "
            code_execution_text += f"スキップ: {code_execution.skipped}\n"
            if code_execution.failed > 0:
                code_execution_text += "\n### 実行エラー:\n"
                for result in code_execution.results:
                    if not result.get("success"):
                        code_execution_text += f"- [ブロック{result.get('block_index', '?')}] [{result.get('language', '?')}] "
                        if result.get("timed_out"):
                            code_execution_text += "タイムアウト\n"
                        else:
                            code_execution_text += f"{result.get('error', 'Unknown error')[:200]}\n"

        prompt = context.prompt_loader.load(
            "review",
            article_title=article.frontmatter.title,
            article_content=article_content,
        )

        # Append code validation and execution info if present
        if code_validation_text:
            prompt += code_validation_text
        if code_execution_text:
            prompt += code_execution_text

        messages = [
            Message(role="system", content=self._get_system_prompt()),
            Message(role="user", content=prompt),
        ]

        return await context.llm_client.complete_json(messages)

    async def _apply_improvements(
        self,
        article: Article,
        review: dict,
        context: StageContext,
    ) -> Article:
        """Apply review improvements to article.

        Args:
            article: Original article
            review: Review feedback
            context: Pipeline context

        Returns:
            Improved article
        """
        # Combine both improvements and code_issues
        improvements = review.get("improvements", [])
        code_issues = review.get("code_issues", [])
        all_feedback = improvements + code_issues

        if not all_feedback:
            return article

        # Update sections that need improvement
        improved_sections = []

        for i, section in enumerate(article.sections):
            section_feedback = [
                fb for fb in all_feedback
                if fb.get("section_index") == i
            ]

            if section_feedback:
                # Regenerate this section with improvements
                improved_section = await self._improve_section(
                    section, section_feedback, context
                )
                improved_sections.append(improved_section)
            else:
                improved_sections.append(section)

        return Article(
            frontmatter=article.frontmatter,
            sections=improved_sections,
            references=article.references,
        )

    async def _improve_section(
        self,
        section: ArticleSection,
        feedback_items: list[dict],
        context: StageContext,
    ) -> ArticleSection:
        """Improve a single section based on feedback.

        Args:
            section: Section to improve
            feedback_items: List of feedback items (improvements or code_issues)
            context: Pipeline context

        Returns:
            Improved section
        """
        # Format feedback - handle both 'suggestion' and 'issue' fields
        feedback_lines = []
        for fb in feedback_items:
            issue = fb.get("issue", "")
            suggestion = fb.get("suggestion", "")
            if issue and suggestion:
                feedback_lines.append(f"- 問題: {issue}\n  改善案: {suggestion}")
            elif suggestion:
                feedback_lines.append(f"- {suggestion}")
            elif issue:
                feedback_lines.append(f"- {issue}")

        improvement_text = "\n".join(feedback_lines)

        prompt = f"""以下のセクションを改善してください。

## 現在のセクション
見出し: {section.heading}
内容:
{section.content}

## 改善点
{improvement_text}

## タスク
上記の改善点を反映した新しいセクション内容を生成してください。

JSON形式で回答してください：
```json
{{
  "content": "改善されたセクション内容"
}}
```
"""

        messages = [
            Message(
                role="system",
                content="技術記事の改善を行うアシスタントです。指摘された改善点を反映して、より良い記事にしてください。",
            ),
            Message(role="user", content=prompt),
        ]

        result = await context.llm_client.complete_json(messages)

        return ArticleSection(
            heading=section.heading,
            level=section.level,
            content=result.get("content", section.content),
            code_examples=section.code_examples,
        )

    def validate_input(self, input_data: Article) -> bool:
        """Validate that we have a valid article."""
        return isinstance(input_data, Article) and bool(input_data.sections)

    def _get_system_prompt(self) -> str:
        """Get system prompt for review."""
        return """あなたは技術記事の編集者です。
与えられた記事をレビューし、改善点を指摘してください。

以下の観点でレビューしてください：
1. 技術的正確性 - 内容は正確か
2. 可読性 - 初心者にもわかりやすいか
3. 構成 - 論理的な流れになっているか
4. コードの品質 - コードは動作するか、ベストプラクティスに従っているか
5. 実用性 - 読者が実践できる内容か

必ずJSON形式で回答してください。"""
