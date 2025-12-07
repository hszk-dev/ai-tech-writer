"""Code validation utilities."""

import ast
import json
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

import yaml

from .languages import get_language_config


@dataclass
class ValidationResult:
    """Result of code validation."""

    valid: bool
    language: str
    error_message: Optional[str] = None
    line_number: Optional[int] = None
    suggestions: list[str] | None = None


class CodeValidator:
    """Validates code blocks for syntax errors."""

    def validate(self, code: str, language: str) -> ValidationResult:
        """Validate code syntax.

        Args:
            code: Code to validate
            language: Programming language

        Returns:
            ValidationResult with validation status
        """
        config = get_language_config(language)

        if config is None:
            # Unknown language, can't validate
            return ValidationResult(
                valid=True,
                language=language,
                suggestions=["Unknown language, skipping syntax check"],
            )

        checker = config.syntax_checker
        if checker is None:
            return ValidationResult(
                valid=True,
                language=language,
                suggestions=[f"No syntax checker available for {language}"],
            )

        # Dispatch to appropriate checker
        if checker == "ast":
            return self._validate_python(code, language)
        elif checker == "acorn":
            return self._validate_javascript(code, language)
        elif checker == "typescript":
            return self._validate_typescript(code, language)
        elif checker == "json":
            return self._validate_json(code, language)
        elif checker == "yaml":
            return self._validate_yaml(code, language)
        elif checker == "bash":
            return self._validate_bash(code, language)
        else:
            return ValidationResult(
                valid=True,
                language=language,
                suggestions=[f"Unknown checker: {checker}"],
            )

    def _validate_python(self, code: str, language: str) -> ValidationResult:
        """Validate Python code using ast.parse."""
        try:
            ast.parse(code)
            return ValidationResult(valid=True, language=language)
        except SyntaxError as e:
            return ValidationResult(
                valid=False,
                language=language,
                error_message=str(e.msg) if e.msg else "Syntax error",
                line_number=e.lineno,
                suggestions=self._get_python_suggestions(e),
            )

    def _get_python_suggestions(self, error: SyntaxError) -> list[str]:
        """Get suggestions for Python syntax errors."""
        suggestions = []
        msg = str(error.msg).lower() if error.msg else ""

        if "indent" in msg:
            suggestions.append("Check indentation - Python uses spaces (4 spaces recommended)")
        if "unexpected eof" in msg:
            suggestions.append("Check for missing closing brackets or quotes")
        if "invalid syntax" in msg:
            suggestions.append("Check for missing colons after if/for/def/class statements")
        if "f-string" in msg:
            suggestions.append("Check f-string syntax - use {variable} inside f'...'")

        return suggestions

    def _validate_javascript(self, code: str, language: str) -> ValidationResult:
        """Validate JavaScript code using basic checks."""
        # Simple heuristic checks (full validation would need Node.js)
        errors = []

        # Check for common issues
        if code.count("{") != code.count("}"):
            errors.append("Mismatched curly braces")
        if code.count("(") != code.count(")"):
            errors.append("Mismatched parentheses")
        if code.count("[") != code.count("]"):
            errors.append("Mismatched square brackets")

        # Check for unclosed strings (simple check)
        single_quotes = len(re.findall(r"(?<!\\)'", code))
        double_quotes = len(re.findall(r'(?<!\\)"', code))
        backticks = len(re.findall(r"(?<!\\)`", code))

        if single_quotes % 2 != 0:
            errors.append("Unclosed single quote string")
        if double_quotes % 2 != 0:
            errors.append("Unclosed double quote string")
        if backticks % 2 != 0:
            errors.append("Unclosed template literal")

        if errors:
            return ValidationResult(
                valid=False,
                language=language,
                error_message="; ".join(errors),
                suggestions=["Check bracket and quote matching"],
            )

        return ValidationResult(valid=True, language=language)

    def _validate_typescript(self, code: str, language: str) -> ValidationResult:
        """Validate TypeScript code (uses same basic checks as JS)."""
        return self._validate_javascript(code, language)

    def _validate_json(self, code: str, language: str) -> ValidationResult:
        """Validate JSON syntax."""
        try:
            json.loads(code)
            return ValidationResult(valid=True, language=language)
        except json.JSONDecodeError as e:
            return ValidationResult(
                valid=False,
                language=language,
                error_message=e.msg,
                line_number=e.lineno,
                suggestions=[
                    "Check for trailing commas (not allowed in JSON)",
                    "Ensure all keys are double-quoted strings",
                    "Check for missing colons or commas",
                ],
            )

    def _validate_yaml(self, code: str, language: str) -> ValidationResult:
        """Validate YAML syntax."""
        try:
            yaml.safe_load(code)
            return ValidationResult(valid=True, language=language)
        except yaml.YAMLError as e:
            error_msg = str(e)
            line_num = None
            if hasattr(e, "problem_mark") and e.problem_mark:
                line_num = e.problem_mark.line + 1

            return ValidationResult(
                valid=False,
                language=language,
                error_message=error_msg[:200],
                line_number=line_num,
                suggestions=[
                    "Check indentation (YAML uses spaces, not tabs)",
                    "Ensure colons have a space after them",
                    "Quote strings with special characters",
                ],
            )

    def _validate_bash(self, code: str, language: str) -> ValidationResult:
        """Validate bash script syntax."""
        # Try using bash -n for syntax check if available
        try:
            result = subprocess.run(
                ["bash", "-n", "-c", code],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return ValidationResult(valid=True, language=language)
            else:
                return ValidationResult(
                    valid=False,
                    language=language,
                    error_message=result.stderr.strip()[:200],
                    suggestions=["Check for missing quotes or semicolons"],
                )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # bash not available or timeout
            return ValidationResult(
                valid=True,
                language=language,
                suggestions=["Could not verify bash syntax (bash not available)"],
            )


def extract_code_blocks(markdown: str) -> list[tuple[str, str]]:
    """Extract code blocks from markdown.

    Args:
        markdown: Markdown content

    Returns:
        List of (language, code) tuples
    """
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, markdown, re.DOTALL)
    return [(lang.strip() or "text", code.strip()) for lang, code in matches]


def validate_code_blocks(markdown: str) -> list[ValidationResult]:
    """Validate all code blocks in markdown.

    Args:
        markdown: Markdown content

    Returns:
        List of ValidationResults for each code block
    """
    validator = CodeValidator()
    blocks = extract_code_blocks(markdown)

    results = []
    for language, code in blocks:
        if language.lower() in ("text", "plaintext", ""):
            continue
        result = validator.validate(code, language)
        results.append(result)

    return results
