#!/usr/bin/env python3
"""
Integration tests for LLM model compatibility.

Tests that all configured models work correctly with DSPy and the LLM client.
Run with: pytest tests/integration/test_llm_models.py -v
"""

import os
import pytest

# Skip all tests if no API key
pytestmark = [
    pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
    pytest.mark.integration
]


class TestDSPyLMCreation:
    """Test DSPy LM creation for various models (no API calls)."""
    
    def test_gpt5_mini_reasoning_params(self):
        """gpt-5-mini requires temperature=1.0 and max_tokens >= 16000."""
        import dspy
        lm = dspy.LM('openai/gpt-5-mini', temperature=1.0, max_tokens=16000)
        assert lm is not None
    
    def test_gpt5_reasoning_params(self):
        """gpt-5 requires temperature=1.0 and max_tokens >= 16000."""
        import dspy
        lm = dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)
        assert lm is not None
    
    def test_gpt4o_mini_any_params(self):
        """gpt-4o-mini works with any temperature."""
        import dspy
        lm = dspy.LM('openai/gpt-4o-mini', temperature=0.5)
        assert lm is not None
    
    def test_reasoning_model_fails_without_params(self):
        """Reasoning models fail without proper params."""
        import dspy
        with pytest.raises(ValueError, match="reasoning models require"):
            dspy.LM('openai/gpt-5-mini', temperature=0.5)


class TestCallDspy:
    """Test call_dspy wrapper (makes API calls)."""
    
    def test_gpt5_mini(self):
        """call_dspy works with gpt-5-mini."""
        from smartmemory.utils.llm_client.dspy import call_dspy
        result = call_dspy(
            model='gpt-5-mini',
            messages=[{'role': 'user', 'content': 'Say test'}],
            max_output_tokens=50
        )
        assert result and len(result) > 0
    
    def test_gpt4o_mini(self):
        """call_dspy works with gpt-4o-mini."""
        from smartmemory.utils.llm_client.dspy import call_dspy
        result = call_dspy(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': 'Say test'}],
            max_output_tokens=50,
            temperature=0.0
        )
        assert result and len(result) > 0
    
    def test_reasoning_model_overrides_temperature(self):
        """call_dspy overrides temperature for reasoning models."""
        from smartmemory.utils.llm_client.dspy import call_dspy
        # Should work even with temperature=0.0 because it gets overridden
        result = call_dspy(
            model='gpt-5-mini',
            messages=[{'role': 'user', 'content': 'Say test'}],
            max_output_tokens=50,
            temperature=0.0
        )
        assert result and len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
