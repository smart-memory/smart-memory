"""
Utility helpers for LLM interactions used across SmartMemory stages.

Standardized on DSPy for transport. No direct OpenAI SDK usage here.

Primary entrypoints:
- call_llm(...): Generic, reusable LLM caller that invokes DSPy and optionally
  enforces JSON output when a response_model or response_format requests it.
- run_ontology_llm(...): Thin wrapper around call_llm for ontology extraction use.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple, Type, List

from smartmemory.utils import get_config
from smartmemory.utils.llm_client.dspy import call_dspy

logger = logging.getLogger(__name__)


def _model_supports_temperature(model_name: str) -> bool:
    """
    Returns False for models that do not support customizing temperature and only accept the default.

    As of 2025-09, OpenAI o3/o4 reasoning families only support default temperature.
    We conservatively treat model names starting with "o3" or "o4" as not supporting temperature.
    """
    if not isinstance(model_name, str):
        return True
    mn = model_name.lower().strip()
    return not (mn.startswith("o3") or mn.startswith("o4") or mn.startswith("gpt-5"))


def call_llm(
        *,
        model: str,
        # Provide either messages OR (system_prompt + user_content)
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        user_content: Optional[str] = None,
        # JSON enforcement helper (no OpenAI parsing; used only to decide if we should enforce JSON)
        response_model: Optional[Type[Any]] = None,
        json_only_instruction: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,  # e.g., {"type": "json_object"}
        # Common generation controls
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,  # Unused for DSPy transport; kept for signature stability
        # Auth/config
        api_key: Optional[str] = None,
        config_section: Optional[str] = "extractor",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Generic LLM call via DSPy.

    Returns (parsed_result, raw_response_text):
    - parsed_result: dict parsed from JSON when JSON was requested and parse succeeded
    - raw_response_text: raw content from the model (always returned when JSON parse not used or fails)
    """
    # Resolve API key: explicit -> config -> env
    resolved_api_key = api_key
    if not resolved_api_key:
        try:
            section = get_config(config_section or "extractor")
            llm_cfg = section.get("llm") if hasattr(section, "get") else None
            if isinstance(llm_cfg, dict):
                resolved_api_key = llm_cfg.get("openai_api_key")
        except Exception as e:
            logger.debug(f"Failed reading config for API key: {e}")
    if not resolved_api_key:
        resolved_api_key = os.getenv("OPENAI_API_KEY")
    if not resolved_api_key:
        raise ValueError("No OpenAI API key found. Provide api_key, set extractor.llm.openai_api_key in config, or set OPENAI_API_KEY.")

    # Build message list for both paths if not provided
    msgs: List[Dict[str, str]] = messages[:] if messages else []
    if not msgs:
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        if user_content:
            msgs.append({"role": "user", "content": user_content})

    # Build messages for DSPy path
    fb_messages: List[Dict[str, str]] = []
    if system_prompt:
        fb_messages.append({"role": "system", "content": system_prompt})
    if messages:
        fb_messages.extend(messages)
    else:
        if user_content:
            fb_messages.append({"role": "user", "content": user_content})

    # Default to JSON if a response_model was requested and no explicit response_format provided
    rf = response_format
    if rf is None and response_model is not None:
        rf = {"type": "json_object"}

    # If caller provided a strong JSON-only instruction, append it as an extra hint
    if json_only_instruction and isinstance(json_only_instruction, str) and json_only_instruction.strip():
        fb_messages.append({"role": "user", "content": json_only_instruction})

    # If we are requesting JSON object responses, many providers require that at least one
    # message explicitly mentions "json" (case-insensitive). If none of the messages contain
    # the word, inject a minimal user message to satisfy validation while remaining neutral.
    expects_json = rf is not None and rf.get("type") == "json_object"  # type: ignore[union-attr]
    if expects_json:
        has_json_word = any(
            isinstance(m.get("content"), str) and ("json" in m["content"].lower()) for m in fb_messages
        )
        if not has_json_word:
            fb_messages.append({
                "role": "user",
                "content": "return a JSON object",
            })

    try:
        # Use DSPy transport exclusively
        temp_arg = temperature if (temperature is not None and _model_supports_temperature(model)) else None
        if temperature is not None and temp_arg is None and temperature != 1:
            logger.warning(
                f"Model '{model}' does not support custom temperature; requested {temperature}. Using model default instead."
            )

        resp = call_dspy(
            model=model,
            messages=fb_messages,
            max_output_tokens=(max_output_tokens or 2000),
            api_key=resolved_api_key,
            response_format=rf,
            temperature=temp_arg,
            reasoning_effort=reasoning_effort,
        )
    except Exception as e:
        logger.error(f"DSPy request failed: {e}")
        resp = None

    # Attempt JSON parse if we requested/expect JSON
    if isinstance(resp, str) and resp.strip():
        if expects_json:
            try:
                parsed = json.loads(resp)
                return parsed, resp
            except Exception as pe:
                logger.warning(f"Failed JSON parse in fallback: {pe}")
                return None, resp
        # Free-form text fallback
        return None, resp

    return None, resp


def run_ontology_llm(
        *,
        model: str,
        user_content: str,
        response_model: Type[Any],
        system_prompt: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        json_only_instruction: Optional[str] = None,
        api_key: Optional[str] = None,
        config_section: Optional[str] = "extractor",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Ontology-focused wrapper around call_llm. Preserves previous behavior/signature.
    """
    # Enforce JSON object fallback when structured parsing isn't available
    rf = {"type": "json_object"}
    return call_llm(
        model=model,
        system_prompt=system_prompt,
        user_content=user_content,
        response_model=response_model,
        json_only_instruction=json_only_instruction
                              or (
                                      (user_content or "")
                                      + "\n\nReturn ONLY a JSON object with keys 'entities' and 'relations'. "
                                        "Do not include markdown fences or commentary. If none found, return {\"entities\": [], \"relations\": []}."
                              ),
        response_format=rf,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        api_key=api_key,
        config_section=config_section,
    )
