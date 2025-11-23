"""
Entity deduplication utilities.
Uses normalization and lemmatization (via Spacy) to merge duplicate entities.
"""

import logging
import unicodedata
from typing import List, Dict, Any
import spacy
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)

# Global spacy instance (lazy loaded)
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not found, though it should be present
            from spacy.lang.en import English
            _nlp = English()
    return _nlp

def normalize_text(text: str) -> str:
    """Normalize text using NFKC and lowercase."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).strip().lower()

def lemmatize_text(text: str) -> str:
    """
    Lemmatize text to handle plurals/variations.
    e.g., "Apples" -> "apple"
    """
    nlp = _get_nlp()
    doc = nlp(text)
    # Use the lemma of the last token if it's a noun, else keep original
    # This is a heuristic; for full names we might want more complex logic
    return " ".join([token.lemma_ for token in doc])

def deduplicate_entities(entities: List[MemoryItem]) -> List[MemoryItem]:
    """
    Deduplicate a list of entities based on normalized and lemmatized names.
    Merges metadata, preferring longer/more complete information.
    """
    if not entities:
        return []

    # Group by canonical key
    groups: Dict[str, List[MemoryItem]] = {}
    
    for ent in entities:
        name = ent.metadata.get('name') or ent.content
        if not name:
            continue
            
        # Create canonical key: normalized + lemmatized
        # We use lemmatized version for grouping to catch "Apple" vs "Apples"
        norm = normalize_text(name)
        lemma = lemmatize_text(norm)
        
        # Use entity type in key to avoid merging "Apple" (Company) with "Apple" (Fruit)
        # if types are distinct. However, often types are messy ("org" vs "company").
        # For now, we'll be aggressive and merge across types if names match exactly,
        # relying on the fact that they likely refer to the same real-world concept in this context.
        # Or we can include type in key if we want to be conservative.
        # Let's include type but normalize it too.
        etype = normalize_text(ent.metadata.get('entity_type') or 'concept')
        
        # Simple heuristic: map common type variations
        if etype in ('organization', 'org', 'company'): etype = 'org'
        if etype in ('person', 'people', 'human'): etype = 'person'
        
        key = f"{lemma}|{etype}"
        
        if key not in groups:
            groups[key] = []
        groups[key].append(ent)

    deduplicated = []
    
    for key, group in groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
            continue
            
        # Merge group
        # 1. Pick best name (longest usually, or most capitalized in original if we had it)
        # Here we only have the items. Let's pick the one with highest confidence.
        best_item = max(group, key=lambda x: (x.metadata.get('confidence') or 0.0))
        
        # 2. Merge attributes
        merged_attrs = {}
        for item in group:
            attrs = item.metadata
            for k, v in attrs.items():
                if k not in merged_attrs and v:
                    merged_attrs[k] = v
        
        # Update best item with merged attributes
        best_item.metadata.update(merged_attrs)
        deduplicated.append(best_item)
        
        logger.debug(f"Merged {len(group)} entities into '{best_item.content}' (Key: {key})")

    return deduplicated
