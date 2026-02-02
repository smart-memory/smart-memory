"""
Claude CLI - Simple interface for Claude via CLI.

Usage:
    from smartmemory.claude_cli import Claude

    claude = Claude()
    answer = claude("What is 2+2?")

    # With options
    answer = claude.chat([
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"}
    ], model="opus")

    # Structured output
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    person = claude.structured("John is 30 years old", schema=Person)
"""

import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from .models import Response, StructuredResponse, Usage
from .provider import Provider, CLIError, CLINotFoundError, CLITimeoutError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Claude:
    """
    Simple interface to Claude via CLI.

    Uses your existing Claude subscription - no API key needed.

    Args:
        model: Default model ("haiku", "sonnet", "opus"). Default: "haiku"
        timeout: Request timeout in seconds. Default: 120
        command: CLI command to use. Default: "claude"

    Examples:
        # Simple
        claude = Claude()
        answer = claude("What is 2+2?")

        # Chat
        answer = claude.chat([
            {"role": "user", "content": "Hello"}
        ])

        # Structured
        class Info(BaseModel):
            name: str

        info = claude.structured("My name is John", schema=Info)
    """

    def __init__(
        self,
        model: str = "haiku",
        timeout: int = 120,
        command: str = "claude",
    ):
        self.model = model
        self.timeout = timeout
        self._provider = Provider(
            command=command,
            timeout=timeout,
            default_model=model,
        )

    def __call__(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        system: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> str:
        """
        Simple completion - returns just the response text.

        Args:
            prompt: The prompt to send
            model: Override default model
            system: Optional system prompt
            timeout: Override default timeout

        Returns:
            Response text as string
        """
        response = self._provider.execute(
            prompt=prompt,
            model=model or self.model,
            system_prompt=system,
            timeout=timeout,
        )

        content = (
            response.get("result")
            or response.get("content")
            or response.get("message")
            or response.get("text")
            or ""
        )

        return content.strip() if isinstance(content, str) else str(content)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Response:
        """
        Chat completion with message history.

        Args:
            messages: List of {"role": "user"|"system"|"assistant", "content": "..."}
            model: Override default model
            timeout: Override default timeout

        Returns:
            Response object with content, model, usage, etc.
        """
        result = self._provider.chat(
            messages=messages,
            model=model or self.model,
            timeout=timeout,
        )

        usage = None
        if result.get("usage"):
            usage = Usage(
                prompt_tokens=result["usage"].get("input_tokens", 0),
                completion_tokens=result["usage"].get("output_tokens", 0),
            )

        return Response(
            content=result["content"],
            model=result["model"],
            usage=usage,
            stop_reason=result.get("stop_reason"),
            session_id=result.get("session_id"),
            raw=result.get("raw"),
        )

    def structured(
        self,
        prompt: str,
        *,
        schema: Type[T],
        model: Optional[str] = None,
        system: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> T:
        """
        Get structured output matching a Pydantic schema.

        Args:
            prompt: The prompt to send
            schema: Pydantic model class to parse response into
            model: Override default model
            system: Optional system prompt
            timeout: Override default timeout

        Returns:
            Instance of the schema class

        Raises:
            ValidationError: If response doesn't match schema
            CLIError: If CLI execution fails
        """
        # Build schema description
        schema_json = json.dumps(schema.model_json_schema(), indent=2)

        enhanced_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema_json}

Return ONLY the JSON object, no other text or markdown."""

        # Combine with system prompt if provided
        full_system = system or ""
        if full_system:
            full_system += "\n\n"
        full_system += "You must respond with valid JSON only. No explanations or markdown."

        response = self._provider.execute(
            prompt=enhanced_prompt,
            model=model or self.model,
            system_prompt=full_system.strip() or None,
            timeout=timeout,
        )

        # Extract content
        content = (
            response.get("result")
            or response.get("content")
            or response.get("message")
            or response.get("text")
            or ""
        )

        if isinstance(content, str):
            content = content.strip()

        # Parse JSON from response
        parsed = self._extract_json(content)

        # Validate against schema
        return schema.model_validate(parsed)

    def structured_safe(
        self,
        prompt: str,
        *,
        schema: Type[T],
        model: Optional[str] = None,
        system: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> StructuredResponse:
        """
        Get structured output with error handling (doesn't raise).

        Returns StructuredResponse with success=False on error instead of raising.
        """
        try:
            data = self.structured(
                prompt=prompt,
                schema=schema,
                model=model,
                system=system,
                timeout=timeout,
            )
            return StructuredResponse(
                data=data,
                content=data.model_dump_json(),
                model=model or self.model,
                success=True,
            )
        except ValidationError as e:
            return StructuredResponse(
                data=None,
                content="",
                model=model or self.model,
                success=False,
                error=f"Validation error: {e}",
            )
        except ValueError as e:
            return StructuredResponse(
                data=None,
                content="",
                model=model or self.model,
                success=False,
                error=f"Parse error: {e}",
            )
        except CLIError as e:
            return StructuredResponse(
                data=None,
                content="",
                model=model or self.model,
                success=False,
                error=str(e),
            )

    def _extract_json(self, content: str) -> Dict[str, Any]:
        """Extract JSON from response content."""
        if isinstance(content, dict):
            return content

        content = str(content).strip()

        # Try direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Remove markdown code fences
        if content.startswith("```"):
            lines = content.split("\n")
            # Skip first line (```json) and last line (```)
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            content = content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

        # Find JSON boundaries
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        # Try array
        start = content.find("[")
        end = content.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from response: {content[:200]}...")

    def verify(self) -> str:
        """Verify CLI is available. Returns version string."""
        return self._provider.verify()

    @property
    def is_available(self) -> bool:
        """Check if CLI is available without raising."""
        try:
            self._provider.verify()
            return True
        except CLIError:
            return False


# Convenience aliases
__all__ = [
    "Claude",
    "Response",
    "StructuredResponse",
    "Usage",
    "CLIError",
    "CLINotFoundError",
    "CLITimeoutError",
]
