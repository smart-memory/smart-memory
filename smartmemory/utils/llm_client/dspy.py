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


def _track_dspy_usage(lm: Any, model: str) -> None:
    """
    Extract and track token usage from DSPy LM history.
    """
    tracker = _get_token_tracker()
    if not tracker:
        return
    
    try:
        # DSPy stores call history in lm.history
        history = getattr(lm, 'history', None)
        if not history:
            return
        
        # Get the most recent entry (last call)
        if isinstance(history, list) and history:
            entry = history[-1]
        else:
            return
        
        # Extract usage from entry
        usage = None
        if isinstance(entry, dict):
            usage = entry.get('usage') or entry.get('response', {}).get('usage')
        elif hasattr(entry, 'usage'):
            usage = entry.usage
        
        if usage:
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            if isinstance(usage, dict):
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                total_tokens = usage.get('total_tokens', 0)
            elif hasattr(usage, 'prompt_tokens'):
                prompt_tokens = getattr(usage, 'prompt_tokens', 0)
                completion_tokens = getattr(usage, 'completion_tokens', 0)
                total_tokens = getattr(usage, 'total_tokens', 0)
            
            if total_tokens > 0 or prompt_tokens > 0:
                tracker.track(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=model
                )
                logger.debug(f"Tracked {total_tokens} tokens for {model}")
    except Exception as e:
        # Don't fail the main call if tracking fails
        logger.debug(f"Token tracking failed: {e}")


def _lm_config_changed(current_lm, new_lm) -> bool:
    """Check if the LM configuration has changed and requires reconfiguration."""
    try:
        # Compare basic attributes that would indicate a config change
        current_model = getattr(current_lm, 'model', None)
        new_model = getattr(new_lm, 'model', None)

        current_api_key = getattr(current_lm, 'api_key', None)
        new_api_key = getattr(new_lm, 'api_key', None)

        current_temp = getattr(current_lm, 'temperature', None)
        new_temp = getattr(new_lm, 'temperature', None)

        return (current_model != new_model or
                current_api_key != new_api_key or
                current_temp != new_temp)
    except Exception:
        # If we can't compare, assume it changed to be safe
        return True


def call_dspy(
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_output_tokens: int,
        api_key: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Invoke the target model via DSPy and return text.

    Notes
    - DSPy provides a unified LM interface; we configure it per call for simplicity.
    - If `response_format` is a json_object, we append a minimal instruction to ensure
      the output is JSON.
    - Raises on failure with structured logging.
    """
    if dspy is None:  # pragma: no cover
        logger.error("DSPy import missing", extra={
            "error": "missing_dependency",
            "module": "dspy",
        })
        raise ImportError("dspy package is required to use the DSPy client")

    # Join chat-like messages into one prompt
    joined = "\n\n".join(m.get("content", "") for m in messages)

    # Enforce JSON output if requested
    if response_format and response_format.get("type") == "json_object":
        if "json" not in joined.lower():
            joined = joined + "\n\nReturn ONLY a JSON object."

    # Configure DSPy LM using provider/model spec string
    # Per DSPy docs: dspy.configure(lm=dspy.LM("openai/<model>", api_key=...))
    provider_model = f"openai/{model}" if "/" not in model else model
    
    # Check if this is a reasoning model (o1/o3/o4/gpt-5.x)
    mn = model.lower().strip()
    is_reasoning_model = mn.startswith("o1") or mn.startswith("o3") or mn.startswith("o4") or mn.startswith("gpt-5")
    
    lm_kwargs: Dict[str, Any] = {}
    if api_key:
        lm_kwargs["api_key"] = api_key
    
    if is_reasoning_model:
        # Reasoning models require temperature=1.0 and max_tokens >= 16000
        lm_kwargs["temperature"] = 1.0
        lm_kwargs["max_tokens"] = max(max_output_tokens or 0, 16000)
    else:
        if temperature is not None:
            lm_kwargs["temperature"] = temperature
        if max_output_tokens:
            lm_kwargs["max_tokens"] = max_output_tokens

    try:
        lm = dspy.LM(provider_model, **lm_kwargs)  # type: ignore[attr-defined]

        # Use thread-local storage to prevent DSPy threading issues
        # Only configure DSPy if this thread hasn't been configured yet
        if not hasattr(_thread_local, 'dspy_configured'):
            dspy.settings.configure(lm=lm)  # type: ignore[attr-defined]
            _thread_local.dspy_configured = True
            _thread_local.current_lm = lm
        else:
            # If already configured, check if we need to update the LM
            current_lm = getattr(_thread_local, 'current_lm', None)
            if current_lm is None or _lm_config_changed(current_lm, lm):
                # Reset DSPy configuration for this thread
                dspy.settings.configure(lm=lm)  # type: ignore[attr-defined]
                _thread_local.current_lm = lm

        # Use the LM interface directly to complete the prompt
        # Some versions expose .complete(prompt) -> object with .text/.completion
        # Prefer complete(), else call the LM directly if callable
        if hasattr(lm, "complete"):
            result = lm.complete(joined)  # type: ignore[attr-defined]
        else:
            result = lm(joined)  # type: ignore[misc]
        
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
        logger.error("Empty output from DSPy", extra={
            "provider_model": provider_model,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        })
        raise RuntimeError("Empty output from DSPy")
    except Exception as e:
        logger.exception("DSPy call failed", extra={
            "provider_model": provider_model,
            "max_output_tokens": max_output_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
            "error_type": type(e).__name__,
        })
        raise
