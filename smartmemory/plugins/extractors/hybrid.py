"""
Hybrid local extractor combining GLiNER2 + ReLiK.

- GLiNER2: Fast, accurate entity extraction with custom types
- ReLiK: State-of-the-art relation extraction and entity linking

Best local extraction option - fully private, no API costs.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.memory_item import MemoryItem
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata

logger = logging.getLogger(__name__)

# Lazy load models
_gliner2_model = None
_relik_model = None


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


def _get_relik(model_name: str = "relik-ie/relik-cie-small"):
    """Lazy load ReLiK model."""
    global _relik_model
    if _relik_model is None:
        try:
            from relik import Relik
            logger.info(f"Loading ReLiK model: {model_name}")
            _relik_model = Relik.from_pretrained(model_name)
            logger.info("ReLiK model loaded successfully")
        except ImportError:
            raise ImportError("relik package required. Install with: pip install relik")
    return _relik_model


# Default entity types for GLiNER2 - comprehensive list for better extraction
DEFAULT_ENTITY_TYPES = [
    "person", "organization", "company", "location", "city", "country",
    "date", "time", "event", "product", "concept", "skill", "technology",
    "project", "tool", "framework", "language", "role", "title",
    "feature", "topic", "idea", "goal", "problem", "solution"
]


@dataclass
class HybridExtractorConfig(MemoryBaseModel):
    """Configuration for hybrid GLiNER2 + ReLiK extractor."""
    # Model selection
    gliner2_model: str = "fastino/gliner2-base-v1"
    relik_model: str = "relik-ie/relik-cie-small"
    
    # Entity types
    entity_types: List[str] = None  # Uses DEFAULT_ENTITY_TYPES if None
    use_relik_entities: bool = True  # If True, merge ReLiK entities with GLiNER2
    
    # GLiNER2 tuning
    gliner2_threshold: float = 0.4  # Confidence threshold (0-1), lower = more entities
    
    # ReLiK tuning
    relik_top_k: int = 100  # Number of candidates to retrieve
    relik_window_size: int = 64  # Token window size
    relik_window_stride: int = 32  # Overlap between windows
    
    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = DEFAULT_ENTITY_TYPES.copy()


class HybridExtractor(ExtractorPlugin):
    """
    Hybrid local extractor: GLiNER2 (entities) + ReLiK (relations).
    
    Best local extraction option:
    - Custom entity types with GLiNER2
    - Relation extraction with ReLiK
    - Entity linking to Wikipedia
    - Fully local, no API costs
    - MPS acceleration on Apple Silicon
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        return PluginMetadata(
            name="hybrid",
            version="2.0.0",
            author="SmartMemory Team",
            description="Local extraction: GLiNER2 entities + ReLiK relations",
            plugin_type="extractor",
            dependencies=["gliner2", "relik"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "relation-extraction", "hybrid", "local", "entity-linking"]
        )
    
    def __init__(self, config: Optional[HybridExtractorConfig] = None):
        self.config = config or HybridExtractorConfig()
    
    def extract(self, text: str) -> dict:
        """
        Extract entities with GLiNER2 and relations with ReLiK.
        
        Returns:
            dict: {'entities': [...], 'relations': [...]}
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}
        
        # Step 1: Extract entities with GLiNER2
        gliner2 = _get_gliner2(self.config.gliner2_model)
        gliner_result = gliner2.extract_entities(
            text, 
            self.config.entity_types,
            threshold=self.config.gliner2_threshold
        )
        
        entities = []
        entity_id_map = {}  # name -> item_id for relation mapping
        
        entities_dict = gliner_result.get('entities', {})
        for entity_type, entity_names in entities_dict.items():
            for name in entity_names:
                if not name or not name.strip():
                    continue
                item_id = f"entity_{len(entities)}"
                entity = MemoryItem(
                    item_id=item_id,
                    content=name,
                    memory_type="entity",
                    metadata={
                        "name": name,
                        "entity_type": entity_type,
                        "source": "gliner2"
                    }
                )
                entities.append(entity)
                entity_id_map[name.lower()] = item_id
        
        logger.info(f"GLiNER2 extracted {len(entities)} entities")
        
        # Step 2: Extract relations with ReLiK
        relik = _get_relik(self.config.relik_model)
        relik_output = relik(
            text,
            top_k=self.config.relik_top_k,
            window_size=self.config.relik_window_size,
            window_stride=self.config.relik_window_stride
        )
        
        # Optionally merge ReLiK entities with GLiNER2 entities
        if self.config.use_relik_entities:
            for span in relik_output.spans:
                name = span.text
                if name.lower() not in entity_id_map:
                    item_id = f"entity_{len(entities)}"
                    # Use ReLiK's Wikipedia label if available
                    entity_type = span.label if span.label != "--NME--" else "entity"
                    entity = MemoryItem(
                        item_id=item_id,
                        content=name,
                        memory_type="entity",
                        metadata={
                            "name": name,
                            "entity_type": entity_type,
                            "wikipedia_label": span.label,
                            "source": "relik"
                        }
                    )
                    entities.append(entity)
                    entity_id_map[name.lower()] = item_id
        
        # Build relations from ReLiK triplets
        relations = self._build_relations_from_relik(relik_output.triplets, entity_id_map, entities)
        
        logger.info(f"ReLiK extracted {len(relations)} relations")
        logger.info(f"Hybrid extractor: {len(entities)} entities, {len(relations)} relations")
        
        return {
            "entities": entities,
            "relations": relations
        }
    
    def _build_relations_from_relik(
        self, 
        triplets: List[Any], 
        entity_id_map: Dict[str, str],
        entities: List[MemoryItem]
    ) -> List[Dict[str, Any]]:
        """
        Build relations from ReLiK triplets, mapping to entities.
        Creates new entities for subjects/objects not found in existing entities.
        """
        relations = []
        
        for triplet in triplets:
            # ReLiK triplets have .subject, .label, .object attributes
            subject = triplet.subject.text.strip()
            obj = triplet.object.text.strip()
            relation = triplet.label.strip()
            confidence = getattr(triplet, 'confidence', 1.0)
            
            if not subject or not obj:
                continue
            
            # Try to map to existing entities
            source_id = self._find_entity_id(subject, entity_id_map, entities)
            target_id = self._find_entity_id(obj, entity_id_map, entities)
            
            # If entities not found, create them
            if not source_id:
                source_id = f"entity_{len(entities)}"
                entities.append(MemoryItem(
                    item_id=source_id,
                    content=subject,
                    memory_type="entity",
                    metadata={"name": subject, "entity_type": "entity", "source": "relik"}
                ))
                entity_id_map[subject.lower()] = source_id
            
            if not target_id:
                target_id = f"entity_{len(entities)}"
                entities.append(MemoryItem(
                    item_id=target_id,
                    content=obj,
                    memory_type="entity",
                    metadata={"name": obj, "entity_type": "entity", "source": "relik"}
                ))
                entity_id_map[obj.lower()] = target_id
            
            relations.append({
                'source_id': source_id,
                'target_id': target_id,
                'relation_type': relation.upper().replace(' ', '_'),
                'confidence': confidence,
                'source': 'relik'
            })
        
        return relations
    
    def _find_entity_id(
        self, 
        name: str, 
        entity_id_map: Dict[str, str],
        entities: List[MemoryItem]
    ) -> Optional[str]:
        """Find entity ID by name (fuzzy match)."""
        name_lower = name.lower()
        
        # Exact match
        if name_lower in entity_id_map:
            return entity_id_map[name_lower]
        
        # Partial match
        for entity_name, entity_id in entity_id_map.items():
            if name_lower in entity_name or entity_name in name_lower:
                return entity_id
        
        return None


# Backwards compatibility alias
HybridGlinerRebelExtractor = HybridExtractor


# Factory function for plugin discovery
def create_extractor(config: Optional[Dict[str, Any]] = None) -> HybridExtractor:
    """Create a hybrid extractor instance."""
    if config:
        extractor_config = HybridExtractorConfig(**config)
        return HybridExtractor(config=extractor_config)
    return HybridExtractor()
