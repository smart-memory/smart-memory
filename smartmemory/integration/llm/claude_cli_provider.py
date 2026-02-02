"""
Claude CLI Provider - Experimental

Uses Claude Code/CLI subprocess for LLM operations without requiring an API key.
Leverages the user's existing Claude subscription authentication.

WARNING: This is experimental and intended for internal testing only.
- Not recommended for production use
- Subject to Claude CLI rate limits and availability
- Output parsing may be fragile

Based on patterns from openclaw/cli-backends.ts
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Type

from .providers import BaseLLMProvider

logger = logging.getLogger(__name__)


# Model aliases matching Claude CLI conventions
CLAUDE_MODEL_ALIASES: Dict[str, str] = {
    "opus": "opus",
    "opus-4.5": "opus",
    "opus-4": "opus",
    "claude-opus-4-5": "opus",
    "claude-opus-4": "opus",
    "sonnet": "sonnet",
    "sonnet-4.5": "sonnet",
    "sonnet-4.1": "sonnet",
    "sonnet-4.0": "sonnet",
    "claude-sonnet-4-5": "sonnet",
    "claude-sonnet-4-1": "sonnet",
    "claude-sonnet-4-0": "sonnet",
    "haiku": "haiku",
    "haiku-3.5": "haiku",
    "claude-haiku-3-5": "haiku",
    # Default mappings
    "default": "sonnet",
    "fast": "haiku",
    "best": "opus",
}

# Environment variables to clear so CLI uses subscription auth
CLEAR_ENV_VARS = ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_OLD"]


class ClaudeCLIProvider(BaseLLMProvider):
    """
    Claude CLI provider - uses subprocess to call claude command.

    This provider requires:
    - Claude Code CLI installed and authenticated (`claude` command available)
    - User logged in to Claude subscription

    It does NOT require:
    - ANTHROPIC_API_KEY environment variable
    """

    def __init__(self, config=None, **kwargs):
        """Initialize Claude CLI provider.

        Args:
            config: Configuration object (optional, CLI uses defaults)
            command: CLI command to use (default: "claude")
            timeout_seconds: Timeout for CLI operations (default: 120)
        """
        self.command = kwargs.pop("command", "claude")
        self.timeout_seconds = kwargs.pop("timeout_seconds", 120)
        self.default_model = kwargs.pop("default_model", "sonnet")

        # Don't call parent __init__ since we don't need API key
        self.config = config
        self.provider = "claude-cli"
        self.api_key = None  # Not used
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._client = None

        # Verify CLI is available
        self._verify_cli_available()

    def _verify_cli_available(self):
        """Check if claude CLI is available and authenticated."""
        try:
            result = subprocess.run(
                [self.command, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError(f"Claude CLI not working: {result.stderr}")
            self.logger.info(f"Claude CLI available: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                f"Claude CLI not found. Install with: npm install -g @anthropic-ai/claude-code\n"
                f"Or use: pip install anthropic && anthropic login"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI timed out checking version")

    def _initialize_client(self, **_kwargs):
        """No client initialization needed for CLI provider."""
        pass

    def _get_clean_env(self) -> Dict[str, str]:
        """Get environment with API keys cleared so CLI uses subscription auth."""
        env = os.environ.copy()
        for key in CLEAR_ENV_VARS:
            env.pop(key, None)
        return env

    def _normalize_model(self, model: str) -> str:
        """Normalize model name to CLI-compatible format."""
        model_lower = model.lower().strip()
        return CLAUDE_MODEL_ALIASES.get(model_lower, model_lower)

    def _build_cli_args(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        json_output: bool = True,
    ) -> List[str]:
        """Build CLI arguments for claude command."""
        args = [self.command]

        # Add prompt flag
        args.extend(["-p", prompt])

        # Output format
        if json_output:
            args.extend(["--output-format", "json"])

        # Skip permission prompts for non-interactive use
        args.append("--dangerously-skip-permissions")

        # Model selection
        if model:
            normalized_model = self._normalize_model(model)
            args.extend(["--model", normalized_model])

        # System prompt
        if system_prompt:
            args.extend(["--append-system-prompt", system_prompt])

        return args

    def _parse_json_response(self, stdout: str) -> Dict[str, Any]:
        """Parse JSON response from CLI output."""
        try:
            # Try direct JSON parse
            return json.loads(stdout.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in output (may have other text)
        lines = stdout.strip().split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue

        # Look for JSON object boundaries
        start = stdout.find("{")
        end = stdout.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(stdout[start:end])
            except json.JSONDecodeError:
                pass

        # Return raw text as content
        return {"result": stdout.strip(), "is_error": False}

    def _run_cli(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        json_output: bool = True,
    ) -> Dict[str, Any]:
        """Execute claude CLI and return parsed response."""
        args = self._build_cli_args(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            json_output=json_output,
        )

        self.logger.debug(f"CLI exec: {' '.join(args[:6])}...")

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=self._get_clean_env(),
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip() or "CLI failed"
                self.logger.error(f"Claude CLI failed: {error_msg}")
                raise RuntimeError(f"Claude CLI error: {error_msg}")

            stdout = result.stdout.strip()
            self.logger.debug(f"CLI response: {len(stdout)} chars")

            if json_output:
                return self._parse_json_response(stdout)
            return {"result": stdout, "is_error": False}

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Claude CLI timed out after {self.timeout_seconds}s")

    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Generate chat completion using Claude CLI.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (opus, sonnet, haiku)
            max_tokens: Ignored (CLI manages this)
            temperature: Ignored (CLI uses defaults)

        Returns:
            Response dict with content, model, usage, metadata
        """
        model = kwargs.get("model", self.default_model)

        # Extract system message and build prompt
        system_prompt = None
        user_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Format as "role: content"
                user_messages.append(f"{msg['role']}: {msg['content']}")

        # Join messages into prompt
        prompt = "\n\n".join(user_messages)

        # Run CLI
        response = self._run_cli(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            json_output=True,
        )

        # Extract content from response
        content = response.get("result", "")
        if not content and "content" in response:
            content = response["content"]
        if not content and "message" in response:
            content = response["message"]
        if not content and "text" in response:
            content = response["text"]
        if not content:
            # Just use the whole response as content
            content = json.dumps(response) if isinstance(response, dict) else str(response)

        # Extract usage if available
        usage = response.get("usage", {})

        return {
            "content": content,
            "models": f"claude-cli:{self._normalize_model(model)}",  # Match existing provider convention
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            },
            "metadata": {
                "finish_reason": response.get("stop_reason", "stop"),
                "response_id": response.get("id", response.get("session_id", "")),
                "provider": "claude-cli",
            },
        }

    def structured_completion(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[Any],
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Generate structured completion.

        Claude CLI doesn't support native structured output,
        so we add JSON instructions to the prompt and parse the result.
        """
        # Add JSON instruction to messages
        schema_hint = ""
        if hasattr(response_model, "model_json_schema"):
            schema_hint = f"\n\nRespond with valid JSON matching this schema:\n{json.dumps(response_model.model_json_schema(), indent=2)}"
        elif hasattr(response_model, "__annotations__"):
            fields = list(response_model.__annotations__.keys())
            schema_hint = f"\n\nRespond with valid JSON containing these fields: {fields}"

        # Append instruction to last user message
        modified_messages = messages.copy()
        for i in range(len(modified_messages) - 1, -1, -1):
            if modified_messages[i]["role"] == "user":
                modified_messages[i] = {
                    "role": "user",
                    "content": modified_messages[i]["content"] + schema_hint + "\n\nReturn ONLY valid JSON, no other text."
                }
                break

        # Get response
        response = self.chat_completion(modified_messages, **kwargs)
        content = response.get("content", "")

        # Try to parse JSON from response
        try:
            # Find JSON in content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                return {
                    "parsed_data": parsed,
                    "raw_content": content,
                    "models": response.get("models", "claude-cli"),
                }
        except json.JSONDecodeError:
            pass

        return None  # Fallback to caller's JSON parsing

    def get_supported_features(self) -> List[str]:
        """Get list of supported features."""
        return [
            "chat_completion",
            # Note: structured_completion works via JSON prompting, not native
        ]

    def validate_connection(self) -> bool:
        """Validate CLI is working."""
        try:
            self._verify_cli_available()
            return True
        except Exception as e:
            self.logger.error(f"CLI validation failed: {e}")
            return False


def create_claude_cli_provider(
    config=None,
    command: str = "claude",
    timeout_seconds: int = 120,
    default_model: str = "sonnet",
) -> ClaudeCLIProvider:
    """Factory function to create Claude CLI provider.

    Args:
        config: Configuration object (optional)
        command: CLI command to use (default: "claude")
        timeout_seconds: Timeout for operations (default: 120)
        default_model: Default model (default: "sonnet")

    Returns:
        ClaudeCLIProvider instance

    Raises:
        RuntimeError: If Claude CLI is not available
    """
    return ClaudeCLIProvider(
        config=config,
        command=command,
        timeout_seconds=timeout_seconds,
        default_model=default_model,
    )
