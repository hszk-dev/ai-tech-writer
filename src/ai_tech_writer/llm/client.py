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
    ) -> CompletionResponse:
        """Generate a completion.

        Args:
            messages: List of chat messages
            config: Optional completion configuration

        Returns:
            CompletionResponse with the generated content
        """
        if config is None:
            config = CompletionConfig(
                model=self.default_model,
                temperature=self.default_temperature,
                max_tokens=self.default_max_tokens,
            )

        response = await acompletion(
            model=config.model,
            messages=[m.to_dict() for m in messages],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p,
            stop=config.stop,
        )

        return CompletionResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=dict(response.usage) if response.usage else {},
            finish_reason=response.choices[0].finish_reason or "",
        )

    async def complete_json(
        self,
        messages: list[Message],
        config: Optional[CompletionConfig] = None,
    ) -> dict[str, Any]:
        """Generate a completion and parse as JSON.

        Args:
            messages: List of chat messages (should instruct JSON output)
            config: Optional completion configuration

        Returns:
            Parsed JSON as dict

        Raises:
            json.JSONDecodeError: If response is not valid JSON
        """
        response = await self.complete(messages, config)
        content = response.content.strip()

        # Handle markdown code blocks
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        return json.loads(content.strip())

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
