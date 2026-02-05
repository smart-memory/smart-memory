from .llm import LLMExtractor
from .llm_single import LLMSingleExtractor, GroqExtractor
from .gliner2 import GLiNER2Extractor
from .hybrid import HybridExtractor
from .reasoning import ReasoningExtractor
from .spacy import SpacyExtractor

# Deprecated - kept for backwards compatibility
from .relik import RelikExtractor

__all__ = [
    # Primary extractors
    'GroqExtractor',          # Default: Groq Llama-3.3 (100% E-F1, 89.3% R-F1, 878ms)
    'LLMExtractor',           # Pure LLM (two-call for precision)
    'LLMSingleExtractor',     # Fast LLM (single-call, configurable model)
    'HybridExtractor',        # Parallel local + LLM
    'GLiNER2Extractor',       # Fast local entities + co-occurrence
    'ReasoningExtractor',     # System 2: Chain-of-thought traces
    'SpacyExtractor',         # Fallback: no API keys needed

    # Deprecated
    'RelikExtractor',
]
