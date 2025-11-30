from .llm import LLMExtractor
from .gliner2 import GLiNER2Extractor
from .ensemble import EnsembleExtractor
from .cascading import CascadingExtractor

# Deprecated - kept for backwards compatibility
from .relik import RelikExtractor
from .spacy import SpacyExtractor

__all__ = [
    # Primary extractors
    'EnsembleExtractor',      # Best quality: parallel local + LLM
    'CascadingExtractor',     # Cost-efficient: local first, LLM if needed
    'LLMExtractor',           # Pure LLM
    'GLiNER2Extractor',       # Fast local entities + co-occurrence
    
    # Deprecated
    'RelikExtractor', 
    'SpacyExtractor',
]
