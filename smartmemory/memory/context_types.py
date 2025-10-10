from typing import TypedDict, List, Any

from smartmemory.models.memory_item import MemoryItem


class IngestionContext(TypedDict, total=False):
    main_item: MemoryItem
    classified_types: List[str]
    semantic_entities: List[str]
    semantic_relations: List[Any]
    entity_ids: dict
    links: dict
    enrichment_result: dict
    provenance_candidates: List[str]
