from .llm import LLMExtractor
from .gliner2 import GLiNER2Extractor
from .hybrid import HybridExtractor
from .reasoning import ReasoningExtractor

# Deprecated - kept for backwards compatibility
from .relik import RelikExtractor
from .spacy import SpacyExtractor

__all__ = [
    # Primary extractors
    'HybridExtractor',        # Best quality: parallel local + LLM
    'LLMExtractor',           # Pure LLM
    'GLiNER2Extractor',       # Fast local entities + co-occurrence
    'ReasoningExtractor',     # System 2: Chain-of-thought traces
    
    # Deprecated
    'RelikExtractor', 
    'SpacyExtractor',
]
