"""
GLiNER2-based entity extractor - fast, local, privacy-preserving.

GLiNER2 is a unified multi-task model for:
- Named Entity Recognition (NER)
- Text Classification
- Structured Data Extraction

Advantages over LLM-based extraction:
- CPU-first: No GPU required, fast inference
- Privacy: 100% local processing, no external API calls
- Schema-driven: Define entity types with descriptions for precision
- Cost: No API costs

Reference: https://arxiv.org/abs/2507.18546
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata
from smartmemory.utils.deduplication import deduplicate_entities, get_canonical_key

logger = logging.getLogger(__name__)

# Lazy load GLiNER2 to avoid import overhead
_gliner2_model = None


def _get_gliner2(model_name: str = "fastino/gliner2-base-v1"):
    """Lazy load GLiNER2 model."""
    global _gliner2_model
    if _gliner2_model is None:
        try:
            from gliner2 import GLiNER2
            logger.info(f"Loading GLiNER2 model: {model_name}")
            _gliner2_model = GLiNER2.from_pretrained(model_name)
            logger.info("GLiNER2 model loaded successfully")
        except ImportError:
            raise ImportError("gliner2 package required. Install with: pip install gliner2")
    return _gliner2_model


# Default entity type descriptions for better extraction accuracy
DEFAULT_ENTITY_SCHEMAS = {
    "person": "Names of people, individuals, executives, or human beings",
    "organization": "Companies, corporations, institutions, agencies, or organizations",
    "location": "Places, cities, countries, addresses, or geographic locations",
    "product": "Products, services, software, tools, or offerings",
    "technology": "Technologies, frameworks, programming languages, or technical concepts",
    "event": "Events, conferences, meetings, or occurrences",
    "concept": "Abstract concepts, ideas, topics, or general terms",
    "skill": "Skills, abilities, competencies, or expertise areas",
    "date": "Dates, time periods, or temporal references",
}


@dataclass
class GLiNER2ExtractorConfig(MemoryBaseModel):
    """Configuration for GLiNER2 extractor."""
    model_name: str = "fastino/gliner2-base-v1"
    confidence_threshold: float = 0.5
    entity_types: Optional[Dict[str, str]] = None  # Custom entity type descriptions
    extract_relations: bool = True  # Whether to extract relations between entities
    

class GLiNER2Extractor(ExtractorPlugin):
    """
    Fast, local entity extractor using GLiNER2.
    
    Features:
    - CPU-optimized inference
    - Schema-driven extraction with entity type descriptions
    - Privacy-preserving (no external API calls)
    - Automatic canonical key generation for entity resolution
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="gliner2",
            version="1.0.0",
            author="SmartMemory Team",
            description="Fast local entity extraction using GLiNER2",
            plugin_type="extractor",
            dependencies=["gliner2>=1.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "entity-extraction", "local", "privacy", "cpu"]
        )
    
    def __init__(
        self,
        config: Optional[GLiNER2ExtractorConfig] = None,
        entity_schemas: Optional[Dict[str, str]] = None
    ):
        self.config = config or GLiNER2ExtractorConfig()
        self.entity_schemas = entity_schemas or self.config.entity_types or DEFAULT_ENTITY_SCHEMAS
        self._model = None
    
    def _get_model(self):
        """Get or load the GLiNER2 model."""
        if self._model is None:
            self._model = _get_gliner2(self.config.model_name)
        return self._model
    
    def extract(
        self,
        text: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract entities from text using GLiNER2.
        
        Args:
            text: Input text to extract entities from
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with 'entities' and 'relations' keys
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}
        
        model = self._get_model()
        
        # Extract entities using GLiNER2
        try:
            result = model.extract_entities(text, self.entity_schemas)
            entities_dict = result.get('entities', {})
        except Exception as e:
            logger.error(f"GLiNER2 extraction failed: {e}")
            return {"entities": [], "relations": []}
        
        # Convert to MemoryItem format
        entities = []
        for entity_type, entity_names in entities_dict.items():
            for name in entity_names:
                if not name or not name.strip():
                    continue
                    
                # Generate canonical key for entity resolution
                canonical_key = get_canonical_key(name, entity_type)
                
                entity = MemoryItem(
                    content=name,
                    memory_type=entity_type,
                    metadata={
                        'name': name,
                        'entity_type': entity_type,
                        'confidence': self.config.confidence_threshold,
                        'source': 'gliner2',
                        'canonical_key': canonical_key
                    }
                )
                entities.append(entity)
        
        # Deduplicate entities
        entities = deduplicate_entities(entities)
        
        # Extract relations if enabled
        relations = []
        if self.config.extract_relations and len(entities) >= 2:
            relations = self._extract_relations(text, entities, model)
        
        logger.info(f"GLiNER2 extracted {len(entities)} entities, {len(relations)} relations")
        
        return {
            "entities": entities,
            "relations": relations
        }
    
    def _extract_relations(
        self,
        text: str,
        entities: List[MemoryItem],
        model
    ) -> List[Dict[str, Any]]:
        """
        Extract relations between entities.
        
        GLiNER2 works best with ontology-defined schemas. Without ontology,
        we only use co-occurrence based relations.
        """
        relations = []
        
        # Build entity name to entity mapping
        entity_names = {e.metadata.get('name', e.content): e for e in entities}
        entity_names_lower = {name.lower(): e for name, e in entity_names.items()}
        
        # NOTE: GLiNER2's extract_json works well ONLY with specific ontology schemas.
        # Generic relation extraction is unreliable. When ontology is provided,
        # the caller should pass relation schemas. For now, skip structured extraction
        # and use co-occurrence only - it's honest about what it can do.
        
        # Fallback: add co-occurrence relations for entity pairs not yet connected
        import re
        sentences = re.split(r'[.!?]+', text)
        
        # Track which pairs already have relations
        connected_pairs = set()
        for rel in relations:
            pair = tuple(sorted([rel['source_id'], rel['target_id']]))
            connected_pairs.add(pair)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            entities_in_sentence = []
            
            for name, entity in entity_names.items():
                if name.lower() in sentence_lower:
                    entities_in_sentence.append(entity)
            
            # Create co-occurrence edges only for unconnected pairs
            for i, e1 in enumerate(entities_in_sentence):
                for e2 in entities_in_sentence[i+1:]:
                    pair = tuple(sorted([e1.item_id, e2.item_id]))
                    if pair not in connected_pairs:
                        connected_pairs.add(pair)
                        relations.append({
                            'source_id': e1.item_id,
                            'target_id': e2.item_id,
                            'relation_type': 'occurs_with',
                            'source': 'gliner2_cooccurrence'
                        })
        
        return relations


# Factory function for plugin discovery
def create_extractor(config: Optional[Dict[str, Any]] = None) -> GLiNER2Extractor:
    """Create a GLiNER2 extractor instance."""
    if config:
        extractor_config = GLiNER2ExtractorConfig(**config)
        return GLiNER2Extractor(config=extractor_config)
    return GLiNER2Extractor()
