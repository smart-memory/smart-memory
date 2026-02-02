"""
Claude CLI Provider - subprocess wrapper for Claude Code CLI.

Executes Claude via subprocess, leveraging user's existing subscription
authentication. No API key required.
"""

import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Model aliases matching Claude CLI conventions
MODEL_ALIASES: Dict[str, str] = {
    # Opus variants
    "opus": "opus",
    "opus-4": "opus",
    "opus-4.5": "opus",
    "claude-opus-4": "opus",
    "claude-opus-4-5": "opus",
    # Sonnet variants
    "sonnet": "sonnet",
    "sonnet-4": "sonnet",
    "sonnet-4.0": "sonnet",
    "sonnet-4.1": "sonnet",
    "sonnet-4.5": "sonnet",
    "claude-sonnet-4": "sonnet",
    "claude-sonnet-4-0": "sonnet",
    "claude-sonnet-4-1": "sonnet",
    "claude-sonnet-4-5": "sonnet",
    # Haiku variants
    "haiku": "haiku",
    "haiku-3.5": "haiku",
    "claude-haiku-3-5": "haiku",
    # Semantic aliases
    "default": "haiku",
    "fast": "haiku",
    "balanced": "sonnet",
    "best": "opus",
    "smart": "opus",
}

# Environment variables to clear so CLI uses subscription auth
CLEAR_ENV_VARS = ["ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_OLD"]


class CLIError(Exception):
    """Error from Claude CLI execution."""

    def __init__(self, message: str, returncode: int = 1, stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


class CLINotFoundError(CLIError):
    """Claude CLI not found."""
    pass


class CLITimeoutError(CLIError):
    """Claude CLI timed out."""
    pass


class Provider:
    """
    Low-level Claude CLI provider.

    Handles subprocess execution, environment setup, and response parsing.
    Use the higher-level `Claude` class for a simpler interface.
    """

    def __init__(
        self,
        command: str = "claude",
        timeout: int = 120,
        default_model: str = "haiku",
    ):
        self.command = command
        self.timeout = timeout
        self.default_model = default_model
        self._verified = False

    def verify(self) -> str:
        """Verify CLI is available. Returns version string."""
        if self._verified:
            return self._version

        try:
            result = subprocess.run(
                [self.command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise CLIError(f"CLI check failed: {result.stderr}", result.returncode)

            self._version = result.stdout.strip()
            self._verified = True
            logger.debug(f"Claude CLI verified: {self._version}")
            return self._version

        except FileNotFoundError:
            raise CLINotFoundError(
                f"Claude CLI not found at '{self.command}'. "
                "Install with: npm install -g @anthropic-ai/claude-code"
            )
        except subprocess.TimeoutExpired:
            raise CLITimeoutError("CLI version check timed out")

    def normalize_model(self, model: str) -> str:
        """Normalize model name to CLI-compatible format."""
        return MODEL_ALIASES.get(model.lower().strip(), model.lower().strip())

    def get_clean_env(self) -> Dict[str, str]:
        """Get environment with API keys cleared."""
        env = os.environ.copy()
        for key in CLEAR_ENV_VARS:
            env.pop(key, None)
        return env

    def build_args(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        json_output: bool = True,
    ) -> List[str]:
        """Build CLI arguments."""
        args = [self.command, "-p", prompt]

        if json_output:
            args.extend(["--output-format", "json"])

        args.append("--dangerously-skip-permissions")

        if model:
            args.extend(["--model", self.normalize_model(model)])

        if system_prompt:
            args.extend(["--append-system-prompt", system_prompt])

        return args

    def parse_response(self, stdout: str) -> Dict[str, Any]:
        """Parse CLI output, extracting JSON if present."""
        stdout = stdout.strip()

        # Try direct JSON parse
        try:
            return json.loads(stdout)
        except json.JSONDecodeError:
            pass

        # Try last line (common pattern)
        lines = stdout.split("\n")
        for line in reversed(lines):
            try:
                return json.loads(line.strip())
            except json.JSONDecodeError:
                continue

        # Find JSON object boundaries
        start = stdout.find("{")
        end = stdout.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(stdout[start:end])
            except json.JSONDecodeError:
                pass

        # Return as plain text
        return {"result": stdout, "_raw_text": True}

    def execute(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        json_output: bool = True,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute CLI and return parsed response."""
        self.verify()

        args = self.build_args(
            prompt=prompt,
            model=model or self.default_model,
            system_prompt=system_prompt,
            json_output=json_output,
        )

        effective_timeout = timeout or self.timeout
        logger.debug(f"Executing: {args[0]} ... (timeout={effective_timeout}s)")

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                env=self.get_clean_env(),
            )

            if result.returncode != 0:
                error = result.stderr.strip() or result.stdout.strip() or "CLI failed"
                raise CLIError(error, result.returncode, result.stderr)

            return self.parse_response(result.stdout)

        except subprocess.TimeoutExpired:
            raise CLITimeoutError(f"CLI timed out after {effective_timeout}s")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute chat completion."""
        system_prompt = None
        user_parts = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_parts.append(f"{msg['role']}: {msg['content']}")

        prompt = "\n\n".join(user_parts)

        response = self.execute(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            timeout=timeout,
        )

        # Normalize response format
        content = (
            response.get("result")
            or response.get("content")
            or response.get("message")
            or response.get("text")
            or json.dumps(response)
        )

        return {
            "content": content,
            "model": f"claude-cli:{self.normalize_model(model or self.default_model)}",
            "usage": response.get("usage", {}),
            "stop_reason": response.get("stop_reason", "stop"),
            "session_id": response.get("session_id") or response.get("id"),
            "raw": response,
        }
