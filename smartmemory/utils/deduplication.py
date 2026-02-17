"""
Entity deduplication and resolution utilities.

Provides:
- Text normalization and lemmatization for entity matching
- Canonical key generation for cross-memory entity resolution
- In-batch deduplication for extraction results
- SemHash-based semantic deduplication
- Singularization for plural handling
"""

import logging
import unicodedata
import warnings
from typing import List, Dict, Any, Tuple, Set

from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)

# Lazy imports for optional dependencies
_nlp = None
_inflect_engine = None
_semhash_available = None

# Type normalization map for consistent entity type matching
TYPE_NORMALIZATION = {
    'organization': 'org',
    'org': 'org',
    'company': 'org',
    'corporation': 'org',
    'business': 'org',
    'person': 'person',
    'people': 'person',
    'human': 'person',
    'individual': 'person',
    'location': 'location',
    'place': 'location',
    'city': 'location',
    'country': 'location',
    'concept': 'concept',
    'idea': 'concept',
    'topic': 'concept',
    'event': 'event',
    'product': 'product',
    'technology': 'technology',
    'tech': 'technology',
    'nationality': 'nationality',
    'language': 'language',
    'award': 'award',
    'work_of_art': 'work_of_art',
    'temporal': 'temporal',
}


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback if model not found
            from spacy.lang.en import English
            _nlp = English()
    return _nlp


def normalize_text(text: str) -> str:
    """Normalize text using NFKC and lowercase."""
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text).strip().lower()


def normalize_entity_type(entity_type: str) -> str:
    """Normalize entity type to canonical form."""
    if not entity_type:
        return "concept"
    normalized = normalize_text(entity_type)
    return TYPE_NORMALIZATION.get(normalized, normalized)


def lemmatize_text(text: str) -> str:
    """
    Lemmatize text to handle plurals/variations.
    e.g., "Apples" -> "apple"
    """
    nlp = _get_nlp()
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def get_canonical_key(name: str, entity_type: str) -> str:
    """
    Generate a canonical key for entity resolution.
    
    This key is used to:
    1. Deduplicate entities within a single extraction
    2. Find existing entities in the graph for cross-memory resolution
    
    Args:
        name: Entity name (e.g., "John Smith", "Apple Inc.")
        entity_type: Entity type (e.g., "person", "organization")
        
    Returns:
        Canonical key string like "john smith|person"
    """
    norm_name = normalize_text(name)

    # Try lemmatization, but fall back to normalized name if it fails or returns empty
    try:
        lemma_name = lemmatize_text(norm_name)
    except Exception:
        lemma_name = ""

    # Use normalized name if lemmatization returned empty/whitespace
    if not lemma_name or not lemma_name.strip():
        lemma_name = norm_name

    norm_type = normalize_entity_type(entity_type)
    return f"{lemma_name}|{norm_type}"


def parse_canonical_key(key: str) -> Tuple[str, str]:
    """Parse a canonical key back into (name, type)."""
    parts = key.split("|", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "concept"


def deduplicate_entities(entities: List[MemoryItem]) -> List[MemoryItem]:
    """
    Deduplicate a list of entities based on canonical keys.
    Merges metadata, preferring higher confidence entries.
    
    Uses get_canonical_key() for consistent key generation across the system.
    """
    if not entities:
        return []

    # Group by canonical key
    groups: Dict[str, List[MemoryItem]] = {}

    for ent in entities:
        name = ent.metadata.get('name') or ent.content
        if not name:
            continue

        entity_type = ent.metadata.get('entity_type') or 'concept'
        key = get_canonical_key(name, entity_type)

        # Store canonical key in metadata for later graph resolution
        ent.metadata['canonical_key'] = key

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


# ============================================================================
# SemHash-based Deduplication
# ============================================================================

def _get_inflect_engine():
    """Lazy load inflect engine for singularization."""
    global _inflect_engine
    if _inflect_engine is None:
        try:
            import inflect
            _inflect_engine = inflect.engine()
        except ImportError:
            logger.warning("inflect not installed, singularization disabled")
            _inflect_engine = False
    return _inflect_engine if _inflect_engine else None


def _is_semhash_available() -> bool:
    """Check if semhash is available."""
    global _semhash_available
    if _semhash_available is None:
        try:
            from semhash import SemHash
            _semhash_available = True
        except ImportError:
            _semhash_available = False
            logger.debug("semhash not installed, semantic deduplication disabled")
    return _semhash_available


def singularize_text(text: str) -> str:
    """
    Singularize text to handle plurals.
    e.g., "Apples" -> "Apple", "companies" -> "company"
    
    Uses inflect library for accurate singularization.
    """
    engine = _get_inflect_engine()
    if not engine:
        return text

    tokens = []
    for tok in text.split():
        singular = engine.singular_noun(tok)
        # singular_noun returns False if not a plural
        tokens.append(singular if isinstance(singular, str) and singular else tok)
    return " ".join(tokens).strip()


class SemHashDeduplicator:
    """
    SemHash-based deduplication for entities and relations.
    
    deduplication approach:
    1. Normalize text (NFKC)
    2. Singularize plurals
    3. Use semantic hashing for similarity matching
    
    This is faster than LLM-based deduplication and works well
    for catching obvious duplicates before LLM clustering.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize SemHash deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering items duplicates (0.0-1.0)
                                  Higher = more strict, fewer matches
        """
        self.threshold = similarity_threshold
        self.original_map: Dict[str, str] = {}  # original -> normalized
        self.items_map: Dict[str, str] = {}  # normalized -> original
        self.duplicates: Dict[str, str] = {}  # duplicate -> canonical

    def deduplicate(self, items: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """
        Deduplicate a list of strings using semantic hashing.
        
        Args:
            items: List of strings to deduplicate
            
        Returns:
            Tuple of (deduplicated items, mapping of duplicate -> canonical)
        """
        if not items:
            return [], {}

        if not _is_semhash_available():
            # Fallback to simple normalization-based dedup
            return self._fallback_deduplicate(items)

        from semhash import SemHash

        # Normalize and singularize each string
        normalized_items = set()
        self.original_map = {}
        self.items_map = {}

        for item in items:
            normalized = normalize_text(item)
            singular = singularize_text(normalized)
            self.original_map[item] = singular
            self.items_map[singular] = item
            normalized_items.add(singular)

        if len(normalized_items) < 2:
            return list(items), {}

        # Deduplicate using SemHash
        try:
            semhash = SemHash.from_records(records=list(normalized_items))
            # Suppress deprecation warning from SemHash about old field names
            # We're already using the new 'selected'/'filtered' fields
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="'deduplicated' and 'duplicates' fields are deprecated")
                result = semhash.self_deduplicate(threshold=self.threshold)

            # Build duplicate mapping
            self.duplicates = {}
            for dup in result.filtered:
                original = dup.record
                if dup.duplicates and len(dup.duplicates) > 0 and len(dup.duplicates[0]) > 0:
                    canonical_normalized = dup.duplicates[0][0]
                    self.items_map[original] = self.items_map.get(canonical_normalized, original)
                    self.duplicates[self.items_map.get(original, original)] = self.items_map.get(canonical_normalized, canonical_normalized)

            # Map back to original strings
            deduplicated = [self.items_map[item] for item in result.selected]

            logger.info(f"SemHash dedup: {len(items)} -> {len(deduplicated)} items "
                        f"({len(items) - len(deduplicated)} duplicates removed)")

            return deduplicated, self.duplicates

        except Exception as e:
            logger.warning(f"SemHash deduplication failed: {e}, using fallback")
            return self._fallback_deduplicate(items)

    def _fallback_deduplicate(self, items: List[str]) -> Tuple[List[str], Dict[str, str]]:
        """Fallback deduplication using simple normalization."""
        seen: Dict[str, str] = {}  # normalized -> original
        duplicates: Dict[str, str] = {}

        for item in items:
            normalized = normalize_text(item)
            singular = singularize_text(normalized)

            if singular in seen:
                duplicates[item] = seen[singular]
            else:
                seen[singular] = item

        return list(seen.values()), duplicates


def semhash_deduplicate_entities(
        entities: List[Dict[str, Any]],
        similarity_threshold: float = 0.95
) -> Tuple[List[Dict[str, Any]], Dict[str, Set[str]]]:
    """
    Deduplicate entities using SemHash.
    
    Args:
        entities: List of entity dicts with 'name' key
        similarity_threshold: SemHash similarity threshold
        
    Returns:
        Tuple of (deduplicated entities, entity_clusters mapping)
    """
    if not entities:
        return [], {}

    # Extract names
    names = []
    name_to_entity: Dict[str, Dict[str, Any]] = {}

    for entity in entities:
        name = entity.get('name') or entity.get('content', '')
        if name:
            names.append(name)
            name_to_entity[name] = entity

    if not names:
        return entities, {}

    # Deduplicate names
    deduplicator = SemHashDeduplicator(similarity_threshold)
    deduplicated_names, duplicates = deduplicator.deduplicate(names)

    # Build entity clusters
    entity_clusters: Dict[str, Set[str]] = {}
    for dup, canonical in duplicates.items():
        if canonical not in entity_clusters:
            entity_clusters[canonical] = {canonical}
        entity_clusters[canonical].add(dup)

    # Build deduplicated entity list
    deduplicated_entities = []
    for name in deduplicated_names:
        if name in name_to_entity:
            entity = dict(name_to_entity[name])
            # Add aliases if this entity has duplicates
            if name in entity_clusters:
                entity['aliases'] = list(entity_clusters[name] - {name})
            deduplicated_entities.append(entity)

    return deduplicated_entities, entity_clusters


def semhash_deduplicate_relations(
        relations: List[str],
        similarity_threshold: float = 0.95
) -> Tuple[List[str], Dict[str, Set[str]]]:
    """
    Deduplicate relation predicates using SemHash.
    
    Args:
        relations: List of relation predicate strings
        similarity_threshold: SemHash similarity threshold
        
    Returns:
        Tuple of (deduplicated predicates, edge_clusters mapping)
    """
    if not relations:
        return [], {}

    deduplicator = SemHashDeduplicator(similarity_threshold)
    deduplicated, duplicates = deduplicator.deduplicate(relations)

    # Build edge clusters
    edge_clusters: Dict[str, Set[str]] = {}
    for dup, canonical in duplicates.items():
        if canonical not in edge_clusters:
            edge_clusters[canonical] = {canonical}
        edge_clusters[canonical].add(dup)

    return deduplicated, edge_clusters
