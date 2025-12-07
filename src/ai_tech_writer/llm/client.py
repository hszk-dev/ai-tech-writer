"""LLM client using LiteLLM for multi-provider support."""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import litellm
from litellm import acompletion


@dataclass
class Message:
    """Chat message."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for API calls."""
        return {"role": self.role, "content": self.content}


@dataclass
class CompletionConfig:
    """Configuration for LLM completion."""

    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stop: Optional[list[str]] = None


@dataclass
class CompletionResponse:
    """Response from LLM completion."""

    content: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""


class LLMClient:
    """Multi-provider LLM client using LiteLLM."""

    def __init__(
        self,
        default_model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ):
        """Initialize LLM client.

        Args:
            default_model: Default model to use (e.g., "claude-sonnet-4-20250514", "gpt-4o")
            temperature: Default temperature for completions
            max_tokens: Default max tokens for completions
        """
        self.default_model = default_model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

        # Disable LiteLLM's verbose logging
        litellm.set_verbose = False

    async def complete(
        self,
        messages: list[Message],
        config: Optional[CompletionConfig] = None,
        response_format: Optional[dict] = None,
    ) -> CompletionResponse:
        """Generate a completion.

        Args:
            messages: List of chat messages
            config: Optional completion configuration
            response_format: Optional response format (e.g., {"type": "json_object"})

        Returns:
            CompletionResponse with the generated content
        """
        if config is None:
            config = CompletionConfig(
                model=self.default_model,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
            )

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
        }

        if config.stop:
            kwargs["stop"] = config.stop

        if response_format:
            kwargs["response_format"] = response_format

        response = await acompletion(**kwargs)

        return CompletionResponse(
            content=response.choices[0].message.content or "",
            model=response.model,
            usage=dict(response.usage) if response.usage else {},
            finish_reason=response.choices[0].finish_reason or "",
        )

    async def complete_json(
        self,
        messages: list[Message],
        config: Optional[CompletionConfig] = None,
        max_retries: int = 3,
        use_json_mode: bool = True,
        schema: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Generate a completion and parse as JSON.

        Args:
            messages: List of chat messages (should instruct JSON output)
            config: Optional completion configuration
            max_retries: Maximum number of retries for JSON parsing failures
            use_json_mode: Whether to use JSON mode (response_format)
            schema: Optional JSON schema to enforce structured output

        Returns:
            Parsed JSON as dict

        Raises:
            json.JSONDecodeError: If response is not valid JSON after retries
        """
        last_error = None

        if config is None:
            config = CompletionConfig(
                model=self.default_model,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
            )

        # For Claude models with schema, use tool_use for structured output
        if schema and self._is_anthropic_model(config.model):
            return await self._complete_with_tool_use(messages, config, schema, max_retries)

        # Build response_format based on model and schema
        response_format = self._build_response_format(config.model, schema, use_json_mode)

        for attempt in range(max_retries):
            try:
                response = await self.complete(messages, config, response_format)
            except Exception:
                # If JSON mode fails (unsupported model), fall back to normal mode
                if use_json_mode and attempt == 0:
                    response_format = None
                    response = await self.complete(messages, config, response_format)
                else:
                    raise
            content = response.content.strip()

            # Try to extract JSON from the response
            parsed = self._extract_json(content)
            if parsed is not None:
                return parsed

            # Store error for potential re-raising
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                last_error = e

            # If not the last attempt, add retry message
            if attempt < max_retries - 1:
                retry_messages = messages + [
                    Message(role="assistant", content=content),
                    Message(
                        role="user",
                        content="JSONのパースに失敗しました。有効なJSON形式で再度回答してください。余計なテキストは含めず、JSONオブジェクトのみを出力してください。",
                    ),
                ]
                messages = retry_messages

        # If all retries failed, raise the last error
        if last_error:
            raise last_error
        raise json.JSONDecodeError("Failed to parse JSON after retries", content, 0)

    def _extract_json(self, content: str) -> Optional[dict[str, Any]]:
        """Extract JSON from content with various strategies.

        Args:
            content: Raw content that may contain JSON

        Returns:
            Parsed JSON dict or None if extraction fails
        """
        import re

        # Strategy 1: Try direct parsing
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Handle markdown code blocks
        cleaned = content
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Find JSON object in text using regex
        json_patterns = [
            r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}",  # Simple nested objects
            r"\{[\s\S]*\}",  # Any content between outermost braces
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        # Strategy 4: Try to find and parse the largest JSON-like block
        start_idx = content.find("{")
        if start_idx != -1:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(content[start_idx:], start_idx):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(content[start_idx : i + 1])
                        except json.JSONDecodeError:
                            break

        return None

    def complete_sync(
        self,
        messages: list[Message],
        config: Optional[CompletionConfig] = None,
    ) -> CompletionResponse:
        """Synchronous version of complete.

        Args:
            messages: List of chat messages
            config: Optional completion configuration

        Returns:
            CompletionResponse with the generated content
        """
        import asyncio

        return asyncio.run(self.complete(messages, config))

    def _is_anthropic_model(self, model: str) -> bool:
        """Check if the model is an Anthropic/Claude model.

        Args:
            model: Model name/ID

        Returns:
            True if model is from Anthropic
        """
        return "claude" in model.lower()

    async def _complete_with_tool_use(
        self,
        messages: list[Message],
        config: CompletionConfig,
        schema: dict,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Generate structured output using Claude's tool_use.

        This method uses a single tool with the desired schema to force
        Claude to output in a specific JSON structure.

        Args:
            messages: List of chat messages
            config: Completion configuration
            schema: JSON schema to enforce
            max_retries: Maximum number of retries

        Returns:
            Parsed JSON as dict matching the schema

        Raises:
            json.JSONDecodeError: If response cannot be parsed
        """
        # Define a tool with the schema
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "structured_output",
                    "description": "Output the result in the specified JSON structure",
                    "parameters": schema,
                },
            }
        ]

        # Force Claude to use this tool
        tool_choice = {"type": "function", "function": {"name": "structured_output"}}

        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                response = await acompletion(**kwargs)

                # Extract tool call arguments
                message = response.choices[0].message

                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_call = message.tool_calls[0]
                    if hasattr(tool_call, "function") and tool_call.function:
                        arguments = tool_call.function.arguments
                        if isinstance(arguments, str):
                            return json.loads(arguments)
                        return arguments

                # Fallback: try to extract from content if no tool call
                if message.content:
                    parsed = self._extract_json(message.content)
                    if parsed is not None:
                        return parsed

            except json.JSONDecodeError as e:
                last_error = e
            except Exception as e:
                # If tool use fails, fall back to regular JSON mode on last attempt
                if attempt == max_retries - 1:
                    raise
                last_error = e

        if last_error:
            raise last_error
        raise json.JSONDecodeError("Failed to extract structured output", "", 0)

    def _build_response_format(
        self,
        model: str,
        schema: Optional[dict],
        use_json_mode: bool,
    ) -> Optional[dict]:
        """Build response_format based on model and schema.

        Args:
            model: Model name/ID
            schema: Optional JSON schema
            use_json_mode: Whether to use JSON mode

        Returns:
            response_format dict or None
        """
        if not use_json_mode:
            return None

        # For OpenAI models with schema, use json_schema format
        if schema and not self._is_anthropic_model(model):
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "output",
                    "strict": True,
                    "schema": schema,
                },
            }

        # Default to basic json_object mode
        return {"type": "json_object"}
