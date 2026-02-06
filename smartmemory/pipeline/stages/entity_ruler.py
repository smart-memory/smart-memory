"""EntityRuler stage — rule-based entity extraction using spaCy NER."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)

# Module-level cache for spaCy model
_nlp_cache: dict = {"nlp": None}

# Label mapping reused from SpacyExtractor._map_spacy_label_to_type()
_LABEL_MAP = {
    "PERSON": "person",
    "ORG": "organization",
    "GPE": "location",
    "LOC": "location",
    "DATE": "temporal",
    "TIME": "temporal",
    "EVENT": "event",
    "PRODUCT": "product",
    "WORK_OF_ART": "work_of_art",
    "FAC": "location",
    "NORP": "organization",
    "LAW": "concept",
    "LANGUAGE": "concept",
    "MONEY": "concept",
    "QUANTITY": "concept",
    "PERCENT": "concept",
    "ORDINAL": "concept",
    "CARDINAL": "concept",
}


def _map_label(label: str) -> str:
    """Map a spaCy NER label to our entity type system."""
    return _LABEL_MAP.get(label.upper(), "concept")


def _get_nlp(model_name: str = "en_core_web_sm"):
    """Lazy-load and cache a spaCy model."""
    if _nlp_cache["nlp"] is not None:
        return _nlp_cache["nlp"]
    try:
        import spacy

        nlp = spacy.load(model_name)
    except Exception:
        nlp = None
    _nlp_cache["nlp"] = nlp
    return nlp


class EntityRulerStage:
    """Extract entities using spaCy NER with rule-based patterns."""

    def __init__(self, nlp=None):
        """Args: nlp — optional pre-loaded spaCy Language (for testing)."""
        self._nlp = nlp

    @property
    def name(self) -> str:
        return "entity_ruler"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        ruler_cfg = config.extraction.entity_ruler
        if not ruler_cfg.enabled:
            return state

        # Build input text from simplified sentences or resolved/raw text
        if state.simplified_sentences:
            text = " ".join(state.simplified_sentences)
        else:
            text = state.resolved_text or state.text

        if not text or not text.strip():
            return state

        nlp = self._nlp or _get_nlp(ruler_cfg.spacy_model)
        if nlp is None:
            logger.warning("spaCy not available, skipping entity ruler")
            return state

        try:
            doc = nlp(text)
            entities: List[Dict[str, Any]] = []
            seen: set = set()

            for ent in doc.ents:
                name = ent.text.strip()
                entity_type = _map_label(ent.label_)
                key = (name.lower(), entity_type)

                if key in seen:
                    continue
                seen.add(key)

                # spaCy NER does not produce per-entity confidence scores.
                # We assign a fixed confidence based on the model quality.
                confidence = 0.9

                if confidence < ruler_cfg.min_confidence:
                    continue

                entities.append(
                    {
                        "name": name,
                        "entity_type": entity_type,
                        "confidence": confidence,
                        "source": "entity_ruler",
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                )

            return replace(state, ruler_entities=entities)
        except Exception as e:
            logger.warning("Entity ruler failed: %s", e)
            return state

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, ruler_entities=[])
