"""
REBEL-based entity and relation extractor.
Extracts entities and relations from text using the REBEL models (via huggingface transformers pipeline).
"""

import re
from typing import Optional
from smartmemory.utils import get_config
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata


class RebelExtractor(ExtractorPlugin):
    """
    REBEL-based entity and relation extractor.
    
    Uses the transformers pipeline with REBEL model to extract triples.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="rebel",
            version="1.0.0",
            author="SmartMemory Team",
            description="Entity and relation extraction using REBEL",
            plugin_type="extractor",
            dependencies=["transformers>=4.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "relation-extraction", "rebel"]
        )
    
    def __init__(self):
        """Initialize the REBEL extractor."""
        self.nlp = None
        self.model_name = None
    
    def _load_model(self):
        """Load REBEL model on demand."""
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers is not installed. Please install with 'pip install transformers'.")
        
        config = get_config('extractor')
        rebel_cfg = config.get('rebel') or {}
        model_name = rebel_cfg.get('model_name')
        if not model_name:
            raise ValueError("No model_name specified in config under extractor['rebel']['model_name'].")
        
        if self.nlp is None or self.model_name != model_name:
            self.nlp = pipeline('text2text-generation', model=model_name)
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
        result = self.nlp(text, max_length=512, clean_up_tokenization_spaces=True)[0]['generated_text']
        
        # Parse REBEL output (subject, relation, object triples)
        triple_pattern = r"\(.*?\)"
        triples_raw = re.findall(triple_pattern, result)
        entities = set()
        relations = []
        
        for triple in triples_raw:
            parts = triple.strip("() ").split(",")
            if len(parts) == 3:
                subj, rel, obj = [p.strip().strip('"') for p in parts]
                entities.add(subj)
                entities.add(obj)
                relations.append((subj, rel, obj))
        
        return {'entities': list(entities), 'relations': relations}
