"""
LiteLLM client adapter.

Provides a thin wrapper around LiteLLM to invoke models across providers with a
single API. Keeps responsibilities narrow: transport only; no JSON parsing here.
"""
from typing import Any, Dict, List, Optional


def call_litellm(
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_output_tokens: int,
        api_key: Optional[str] = None,  # LiteLLM usually reads from env/provider settings
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Invoke the target model via LiteLLM and return text.

    Notes
    - LiteLLM abstracts provider differences; use 'response_format' when supported.
    - Some LiteLLM/provider combos do NOT accept 'reasoning' kwargs; we do NOT pass it.
    - Falls back to standard chat completion behavior otherwise.
    """
    from litellm import completion  # type: ignore

    if completion is None:  # pragma: no cover
        raise ImportError("litellm package is required to use the LiteLLM client")

    base_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_output_tokens,
    }
    if temperature is not None:
        base_kwargs["temperature"] = temperature

    def _extract_content(r: Any) -> Optional[str]:
        if r is None:
            return None
        choices = getattr(r, "choices", None)
        if choices is None and isinstance(r, dict):
            choices = r.get("choices")
        if not choices:
            return None
        first = choices[0]
        msg = getattr(first, "message", None)
        if msg is None and isinstance(first, dict):
            msg = first.get("message")
        if msg is None:
            return None
        # message can be dict-like or object with attribute
        content = None
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        return content if isinstance(content, str) else None

    # Try with all extras
    kwargs = dict(base_kwargs)
    if response_format:
        kwargs["response_format"] = response_format

    try:
        resp = completion(**kwargs)
        text = _extract_content(resp)
        if isinstance(text, str) and text.strip():
            return text
    except TypeError as te:
        print(f"[litellm_client] TypeError calling completion with extras: {te}")
    except Exception as e:
        print(f"[litellm_client] Error calling completion with extras: {e}")

    # Retry minimal (no extras)
    try:
        resp = completion(**base_kwargs)
        text = _extract_content(resp)
        if isinstance(text, str) and text.strip():
            return text
    except Exception as e:
        print(f"[litellm_client] Minimal retry failed: {e}")

    # Last resort: str(resp)
    try:
        s = str(resp)
        return s if isinstance(s, str) and s.strip() else None
    except Exception:
        return None
