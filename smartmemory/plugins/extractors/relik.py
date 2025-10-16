from typing import Optional
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata
from smartmemory.configuration import MemoryConfig


class RelikExtractor(ExtractorPlugin):
    """
    Relik-based relation extractor.
    
    Uses the Relik library to extract entity-relation-entity triples.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="relik",
            version="1.0.0",
            author="SmartMemory Team",
            description="Entity and relation extraction using Relik",
            plugin_type="extractor",
            dependencies=["relik>=0.1.0"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "relation-extraction", "relik"]
        )
    
    def __init__(self):
        """Initialize the Relik extractor."""
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
    
    def extract(self, text: str, user_id: Optional[str] = None) -> dict:
        """
        Extract entities and relations from text.
        
        Args:
            text: The text to extract from
            user_id: Optional user ID for context
        
        Returns:
            dict: Dictionary with 'entities' and 'relations' keys
        """
        self._load_model()
        output = self.model(text)
        triples_raw = output.triples
        entities = list(set([t[0] for t in triples_raw] + [t[2] for t in triples_raw]))
        relations = [(t[0], t[1], t[2]) for t in triples_raw]
        
        return {'entities': entities, 'relations': relations}
