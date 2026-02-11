"""
DSPy client adapter.

A thin wrapper around DSPy (https://github.com/stanfordnlp/dspy) to invoke LLMs
through DSPy's language model interface. Focuses on transport only and returns
raw text; parsing/validation is the caller's responsibility.

Includes automatic token usage tracking via smartmemory.utils.token_tracking.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

try:
    import dspy  # type: ignore
except Exception:  # pragma: no cover
    dspy = None  # type: ignore

# Lazy import for token tracking to avoid circular imports
_token_tracker = None


def _get_token_tracker():
    """Lazy load token tracker."""
    global _token_tracker
    if _token_tracker is None:
        try:
            from smartmemory.utils.token_tracking import get_global_tracker

            _token_tracker = get_global_tracker()
        except ImportError:
            _token_tracker = False  # Mark as unavailable
    return _token_tracker if _token_tracker else None


logger = logging.getLogger(__name__)

# Thread-local storage for DSPy configuration to prevent threading issues
_thread_local = threading.local()


def _extract_dspy_usage(lm: Any) -> Optional[Dict[str, int]]:
    """Extract token usage from DSPy LM history. Returns dict or None."""
    try:
        history = getattr(lm, "history", None)
        if not history:
            return None

        if isinstance(history, list) and history:
            entry = history[-1]
        else:
            return None

        usage = None
        if isinstance(entry, dict):
            usage = entry.get("usage") or entry.get("response", {}).get("usage")
        elif hasattr(entry, "usage"):
            usage = entry.usage

        if not usage:
            return None

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        elif hasattr(usage, "prompt_tokens"):
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)
        else:
            return None

        if total_tokens > 0 or prompt_tokens > 0:
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
    except Exception as e:
        logger.debug(f"Usage extraction failed: {e}")
    return None


def _track_dspy_usage(lm: Any, model: str) -> None:
    """Extract and track token usage from DSPy LM history."""
    usage_dict = _extract_dspy_usage(lm)

    # Store in thread-local for per-request retrieval (CFS-1)
    _thread_local.last_usage = usage_dict

    if not usage_dict:
        return

    tracker = _get_token_tracker()
    if not tracker:
        return

    try:
        tracker.track(
            prompt_tokens=usage_dict["prompt_tokens"],
            completion_tokens=usage_dict["completion_tokens"],
            total_tokens=usage_dict["total_tokens"],
            model=model,
        )
        logger.debug("Tracked %d tokens for %s", usage_dict["total_tokens"], model)
    except Exception as e:
        logger.debug(f"Token tracking failed: {e}")


def get_last_usage() -> Optional[Dict[str, int]]:
    """Return token usage from the most recent ``call_dspy()`` on this thread.

    Returns dict with prompt_tokens, completion_tokens, total_tokens or None.
    Clears the stored value after reading (consume-once).
    """
    usage = getattr(_thread_local, "last_usage", None)
    _thread_local.last_usage = None
    return usage


def _lm_config_changed(current_lm, new_lm) -> bool:
    """Check if the LM configuration has changed and requires reconfiguration."""
    try:
        # Compare basic attributes that would indicate a config change
        current_model = getattr(current_lm, "model", None)
        new_model = getattr(new_lm, "model", None)

        current_api_key = getattr(current_lm, "api_key", None)
        new_api_key = getattr(new_lm, "api_key", None)

        current_temp = getattr(current_lm, "temperature", None)
        new_temp = getattr(new_lm, "temperature", None)

        return current_model != new_model or current_api_key != new_api_key or current_temp != new_temp
    except Exception:
        # If we can't compare, assume it changed to be safe
        return True


def call_dspy(
    *,
    model: str,
    messages: List[Dict[str, str]],
    max_output_tokens: int,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    response_format: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Invoke the target model via DSPy and return text.

    Notes
    - DSPy provides a unified LM interface; we configure it per call for simplicity.
    - If `response_format` is a json_object, we append a minimal instruction to ensure
      the output is JSON.
    - Raises on failure with structured logging.
    """
    if dspy is None:  # pragma: no cover
        logger.error(
            "DSPy import missing",
            extra={
                "error": "missing_dependency",
                "module": "dspy",
            },
        )
        raise ImportError("dspy package is required to use the DSPy client")

    # Join chat-like messages into one prompt
    joined = "\n\n".join(m.get("content", "") for m in messages)

    # Enforce JSON output if requested
    if response_format and response_format.get("type") == "json_object":
        if "json" not in joined.lower():
            joined = joined + "\n\nReturn ONLY a JSON object."

    # Configure DSPy LM using provider/model spec string
    # Per DSPy docs: dspy.configure(lm=dspy.LM("openai/<model>", api_key=...))
    # Detect provider from model name or explicit prefix
    if api_base:
        # Custom API base (e.g., LM Studio, Groq, Cerebras) — always use openai-compatible provider
        # Always prepend openai/ so litellm routes to api_base. If model is "openai/gpt-oss-20b",
        # this becomes "openai/openai/gpt-oss-20b" → litellm strips first prefix and sends
        # "openai/gpt-oss-20b" to the endpoint, which is the correct model name.
        provider_model = f"openai/{model}"
    elif "/" not in model:
        # Auto-detect provider from model name
        if model.startswith("llama") or model.startswith("mixtral"):
            provider_model = f"groq/{model}"
        elif model.startswith("claude"):
            provider_model = f"anthropic/{model}"
        elif model.startswith("gemini"):
            provider_model = f"gemini/{model}"
        else:
            provider_model = f"openai/{model}"
    else:
        provider_model = model

    # Check if this is a reasoning model (o1/o3/o4/gpt-5.x)
    mn = model.lower().strip()
    is_reasoning_model = mn.startswith("o1") or mn.startswith("o3") or mn.startswith("o4") or mn.startswith("gpt-5")

    lm_kwargs: Dict[str, Any] = {}
    if api_key:
        lm_kwargs["api_key"] = api_key
    if api_base:
        lm_kwargs["api_base"] = api_base

    if is_reasoning_model:
        # Reasoning models require temperature=1.0 and max_tokens >= 16000
        lm_kwargs["temperature"] = 1.0
        lm_kwargs["max_tokens"] = max(max_output_tokens or 0, 16000)
    else:
        if temperature is not None:
            lm_kwargs["temperature"] = temperature
        if max_output_tokens:
            lm_kwargs["max_tokens"] = max_output_tokens

    # Pass response_format to litellm for structured output (JSON schema support)
    if response_format:
        lm_kwargs["response_format"] = response_format

    # Pass extra_body for provider-specific params (e.g. enable_thinking for LM Studio)
    if extra_body:
        lm_kwargs["extra_body"] = extra_body

    try:
        lm = dspy.LM(provider_model, **lm_kwargs)  # type: ignore[attr-defined]

        # Use DSPy's context manager to handle thread-safe configuration
        # This avoids the "can only be changed by the thread that initially configured it" error
        with dspy.context(lm=lm):  # type: ignore[attr-defined]
            # DSPy 3.x: Call LM directly with prompt, returns list of strings
            result = lm(joined)  # type: ignore[misc]
            logger.debug(f"DSPy raw result type: {type(result)}, value: {str(result)[:200]}")

        # Track token usage from DSPy history
        _track_dspy_usage(lm, model)

        # Try common attributes to get the text content
        for attr in ("text", "completion", "output_text"):
            val = getattr(result, attr, None)
            # Some backends surface a single-element list of strings
            if isinstance(val, list) and val and isinstance(val[0], str):
                out_text = "\n".join(v for v in val if isinstance(v, str))
                if out_text.strip():
                    return out_text
            if isinstance(val, str) and val.strip():
                return val
        # As a fallback, str(result)
        # Coerce list-like results to string
        if isinstance(result, list) and result and isinstance(result[0], str):
            s = "\n".join(x for x in result if isinstance(x, str))
        else:
            s = str(result)
        if s.strip():
            return s
        # Empty output is an error
        logger.error(
            "Empty output from DSPy",
            extra={
                "provider_model": provider_model,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            },
        )
        raise RuntimeError("Empty output from DSPy")
    except Exception as e:
        logger.exception(
            "DSPy call failed",
            extra={
                "provider_model": provider_model,
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
                "error_type": type(e).__name__,
            },
        )
        raise
