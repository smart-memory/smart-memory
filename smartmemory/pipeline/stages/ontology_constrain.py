"""OntologyConstrain stage — merge, validate, and filter extraction results."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Set

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.graph.ontology_graph import OntologyGraph
    from smartmemory.observability.events import RedisStreamQueue
    from smartmemory.ontology.entity_pair_cache import EntityPairCache
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class OntologyConstrainStage:
    """Merge ruler + LLM entities, validate types against ontology, filter relations."""

    def __init__(
        self,
        ontology_graph: OntologyGraph,
        promotion_queue: RedisStreamQueue | None = None,
        entity_pair_cache: EntityPairCache | None = None,
    ):
        self._ontology = ontology_graph
        self._promotion_queue = promotion_queue
        self._entity_pair_cache = entity_pair_cache

    @property
    def name(self) -> str:
        return "ontology_constrain"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        constrain_cfg = config.extraction.constrain
        promotion_cfg = config.extraction.promotion

        # Step 1: Merge ruler + LLM entities
        merged = self._merge_entities(state.ruler_entities, state.llm_entities)

        # Step 2: Validate entity types against ontology + track frequency
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        promotion_candidates: List[Dict[str, Any]] = []
        accepted_names: Set[str] = set()

        for entity in merged:
            entity_type = self._get_entity_type(entity)
            name = self._get_entity_name(entity)
            confidence = self._get_confidence(entity)
            status = self._ontology.get_type_status(entity_type.title())

            if status in ("seed", "confirmed", "provisional"):
                accepted.append(entity)
                accepted_names.add(name.lower())
                # Track frequency for accepted entities
                self._ontology.increment_frequency(entity_type.title(), confidence)
            elif confidence >= constrain_cfg.confidence_threshold:
                # Unknown type with sufficient confidence — add as provisional
                self._ontology.add_provisional(entity_type.title())
                self._ontology.increment_frequency(entity_type.title(), confidence)
                promotion_candidates.append(entity)
                accepted.append(entity)
                accepted_names.add(name.lower())
            else:
                rejected.append(entity)

        # Step 2b: Inject cached entity-pair relations
        if self._entity_pair_cache and len(accepted) >= 2:
            cached_relations = self._lookup_cached_relations(accepted, state.workspace_id)
            if cached_relations:
                state = replace(state, llm_relations=list(state.llm_relations) + cached_relations)

        # Step 3: Filter relations — keep only those with both endpoints accepted
        valid_relations = self._filter_relations(state.llm_relations, accepted, accepted_names)

        # Step 4: Apply limits
        accepted = accepted[: constrain_cfg.max_entities]
        valid_relations = valid_relations[: constrain_cfg.max_relations]

        # Step 5: Enqueue promotion candidates (async) or auto-promote (fallback)
        if not promotion_cfg.require_approval:
            self._handle_promotion(promotion_candidates)

        return replace(
            state,
            entities=accepted,
            relations=valid_relations,
            rejected=rejected,
            promotion_candidates=promotion_candidates,
        )

    def _handle_promotion(self, candidates: List[Dict[str, Any]]) -> None:
        """Enqueue candidates to promotion stream, or auto-promote as fallback."""
        for candidate in candidates:
            entity_type = self._get_entity_type(candidate)
            entity_name = self._get_entity_name(candidate)
            confidence = self._get_confidence(candidate)

            if self._promotion_queue:
                try:
                    self._promotion_queue.enqueue(
                        {
                            "entity_name": entity_name,
                            "entity_type": entity_type,
                            "confidence": confidence,
                        }
                    )
                    continue
                except Exception as e:
                    logger.debug("Promotion queue unavailable, falling back to inline: %s", e)

            # Fallback: direct promote
            self._ontology.promote(entity_type.title())

    def _lookup_cached_relations(
        self, accepted: List[Dict[str, Any]], workspace_id: str | None
    ) -> List[Dict[str, Any]]:
        """Look up cached entity-pair relations for accepted entities."""
        cached: List[Dict[str, Any]] = []
        ws = workspace_id or "default"
        names = [self._get_entity_name(e) for e in accepted]
        for i, name_a in enumerate(names):
            for name_b in names[i + 1 :]:
                try:
                    relations = self._entity_pair_cache.lookup(name_a, name_b, ws)  # type: ignore[union-attr]
                    if relations:
                        for rel in relations:
                            cached.append(
                                {
                                    "source_id": name_a.lower(),
                                    "target_id": name_b.lower(),
                                    "relation_type": rel.get("relation_type", "RELATED"),
                                    "confidence": rel.get("confidence", 0.5),
                                    "source": "entity_pair_cache",
                                }
                            )
                except Exception:
                    pass
        return cached

    def _merge_entities(
        self,
        ruler_entities: List[Any],
        llm_entities: List[Any],
    ) -> List[Dict[str, Any]]:
        """Merge ruler and LLM entities by name. Higher confidence wins."""
        merged: Dict[str, Dict[str, Any]] = {}

        # Index ruler entities first (preferred type source)
        for entity in ruler_entities:
            name = self._get_entity_name(entity).lower()
            merged[name] = self._normalize_entity(entity, source="ruler")

        # Merge LLM entities
        for entity in llm_entities:
            name = self._get_entity_name(entity).lower()
            normalized = self._normalize_entity(entity, source="llm")
            if name in merged:
                # Keep ruler type, but take higher confidence
                existing = merged[name]
                if normalized.get("confidence", 0) > existing.get("confidence", 0):
                    existing["confidence"] = normalized["confidence"]
            else:
                merged[name] = normalized

        return list(merged.values())

    def _normalize_entity(self, entity: Any, source: str) -> Dict[str, Any]:
        """Convert any entity format to a standard dict."""
        if isinstance(entity, dict):
            result = dict(entity)
            result.setdefault("source", source)
            return result

        # Handle MemoryItem objects from LLM extractor
        name = getattr(entity, "content", "") or ""
        metadata = getattr(entity, "metadata", {}) or {}
        return {
            "name": metadata.get("name", name),
            "entity_type": metadata.get("entity_type", "concept"),
            "confidence": metadata.get("confidence", 0.5),
            "source": source,
            "item_id": getattr(entity, "item_id", None),
        }

    def _get_entity_name(self, entity: Any) -> str:
        """Extract name from dict or MemoryItem."""
        if isinstance(entity, dict):
            return entity.get("name", "")
        metadata = getattr(entity, "metadata", {}) or {}
        return metadata.get("name", getattr(entity, "content", ""))

    def _get_entity_type(self, entity: Any) -> str:
        """Extract entity type from dict or MemoryItem."""
        if isinstance(entity, dict):
            return entity.get("entity_type", "concept")
        metadata = getattr(entity, "metadata", {}) or {}
        return metadata.get("entity_type", "concept")

    def _get_confidence(self, entity: Any) -> float:
        """Extract confidence from dict or MemoryItem."""
        if isinstance(entity, dict):
            return float(entity.get("confidence", 0.5))
        metadata = getattr(entity, "metadata", {}) or {}
        return float(metadata.get("confidence", 0.5))

    def _filter_relations(
        self,
        relations: List[Any],
        accepted_entities: List[Dict[str, Any]],
        accepted_names: Set[str],
    ) -> List[Dict[str, Any]]:
        """Keep only relations where both endpoints are in accepted entities."""
        # Build a set of accepted entity IDs
        accepted_ids: Set[str] = set()
        for entity in accepted_entities:
            eid = entity.get("item_id") or entity.get("name", "").lower()
            accepted_ids.add(eid)

        valid = []
        for rel in relations:
            if not isinstance(rel, dict):
                continue
            source = rel.get("source_id", "")
            target = rel.get("target_id", "")
            if source in accepted_ids and target in accepted_ids:
                valid.append(rel)
        return valid

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, entities=[], relations=[], rejected=[], promotion_candidates=[])
