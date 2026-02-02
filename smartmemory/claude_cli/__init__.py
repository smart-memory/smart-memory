"""
Claude CLI - Simple interface for Claude via CLI subprocess.

No API key required - uses your existing Claude subscription.

Usage:
    from smartmemory.claude_cli import Claude

    # Simple completion
    claude = Claude()
    answer = claude("What is 2+2?")  # "4"

    # With model selection
    answer = claude("Complex task", model="opus")

    # Chat with history
    response = claude.chat([
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"}
    ])
    print(response.content)

    # Structured output
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    person = claude.structured(
        "Extract: John is 30 years old",
        schema=Person
    )
    print(person.name)  # "John"

Models:
    - haiku (default): Fast, efficient
    - sonnet: Balanced
    - opus: Most capable

Aliases:
    - fast -> haiku
    - balanced -> sonnet
    - best/smart -> opus
"""

from .core import Claude
from .models import Response, StructuredResponse, Usage
from .provider import (
    Provider,
    CLIError,
    CLINotFoundError,
    CLITimeoutError,
    MODEL_ALIASES,
)

__all__ = [
    # Main interface
    "Claude",
    # Response types
    "Response",
    "StructuredResponse",
    "Usage",
    # Low-level provider
    "Provider",
    # Errors
    "CLIError",
    "CLINotFoundError",
    "CLITimeoutError",
    # Constants
    "MODEL_ALIASES",
]

__version__ = "0.1.0"
