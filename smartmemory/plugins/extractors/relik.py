"""
DEPRECATED: Standalone ReLiK extractor.

This extractor is deprecated in favor of:
- HybridGlinerRebelExtractor (GLiNER2 + ReLiK) for local extraction
- CascadingExtractor for local + LLM enhancement
- EnsembleExtractor for best quality

ReLiK is now integrated into the hybrid extractor for better results.
"""

import warnings
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata
from smartmemory.configuration import MemoryConfig


class RelikExtractor(ExtractorPlugin):
    """
    DEPRECATED: Standalone ReLiK extractor.
    
    Use HybridGlinerRebelExtractor, CascadingExtractor, or EnsembleExtractor instead.
    ReLiK is integrated into those extractors for better entity + relation extraction.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="relik",
            version="1.0.0",
            author="SmartMemory Team",
            description="[DEPRECATED] Use HybridGlinerRebelExtractor instead",
            plugin_type="extractor",
            dependencies=["relik>=0.1.0"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "relation-extraction", "relik", "deprecated"]
        )
    
    def __init__(self):
        """Initialize the Relik extractor."""
        warnings.warn(
            "RelikExtractor is deprecated. Use HybridGlinerRebelExtractor, "
            "CascadingExtractor, or EnsembleExtractor instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.model = None
        self.model_name = None
    
    def _load_model(self):
        """Load Relik model on demand."""
        try:
            from relik import Relik
        except ImportError:
            raise ImportError("relik is not installed. Please install with 'pip install relik'.")
        
        config = MemoryConfig().extractor
        relik_cfg = config.get('relik') or {}
        model_name = relik_cfg.get('model_name')
        if not model_name:
            raise ValueError("No model_name specified in config under extractor['relik']['model_name'].")
        
        if self.model is None or self.model_name != model_name:
            self.model = Relik.from_pretrained(model_name)
            self.model_name = model_name
    
    def extract(self, text: str) -> dict:
        """
        Extract entities and relations from text.
        
        Args:
            text: The text to extract from
        
        Returns:
            dict: Dictionary with 'entities' and 'relations' keys
        """
        self._load_model()
        output = self.model(text)
        triples_raw = output.triples
        entities = list(set([t[0] for t in triples_raw] + [t[2] for t in triples_raw]))
        relations = [(t[0], t[1], t[2]) for t in triples_raw]
        
        return {'entities': entities, 'relations': relations}
