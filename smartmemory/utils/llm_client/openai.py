"""
OpenAI Responses API client adapter.

Provides a thin wrapper around OpenAI's Responses API to turn a list of chat-like
messages into a single input block and return text output.

Do not perform JSON parsing or post-processing here; return raw text so the
caller can centralize parsing and validation.
"""
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception as e:  # pragma: no cover
    OpenAI = None  # type: ignore


def call_openai_responses(
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_output_tokens: int,
        api_key: str,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Invoke the OpenAI Responses API and return output text.

    Arguments
    - model: OpenAI model name (e.g., gpt-5-mini).
    - messages: list of dicts with keys {role, content}.
    - max_output_tokens: output token budget for Responses API.
    - api_key: OpenAI API key.
    - response_format: optional dict, e.g. {"type": "json_object"}.
    - reasoning_effort: optional string, e.g. "minimal".
    """
    if OpenAI is None:  # pragma: no cover
        raise ImportError("openai package is required to use the Responses API client")

    client = OpenAI(api_key=api_key)

    joined_content = "\n\n".join(m.get("content", "") for m in messages)
    params: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": joined_content}],
        "max_output_tokens": max_output_tokens,
    }
    if reasoning_effort:
        params["reasoning"] = {"effort": reasoning_effort}

    # Some SDK versions don't accept response_format yet; try and fallback
    if response_format:
        try:
            params_with_format = dict(params)
            params_with_format["response_format"] = response_format
            response = client.responses.create(**params_with_format)
        except TypeError:
            response = client.responses.create(**params)
    else:
        response = client.responses.create(**params)

    # Prefer response.output_text, else dig into blocks
    text = getattr(response, "output_text", None)
    if text:
        return text

    try:
        outputs = getattr(response, "output", None) or []
        if outputs and hasattr(outputs[0], "content") and outputs[0].content:
            first = outputs[0].content[0]
            if hasattr(first, "text") and hasattr(first.text, "value"):
                return first.text.value
    except Exception:
        return None

    return None
