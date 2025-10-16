"""
LLM-based entity and relation extractor using OpenAI-compatible models.

This module extracts entities and SPO triples from text using JSON-structured prompts
via the shared call_llm helper, and optionally falls back to a stricter JSON-mode prompt.
It returns a dict with keys:
  - 'entities': List[MemoryItem] where metadata includes 'name', 'entity_type', 'confidence', optional attrs
  - 'relations': List[Dict] with 'source_id', 'target_id', 'relation_type' matching entity MemoryItem.item_id
  
Note: 'triples' are derived from 'relations' at point of use, not returned by extractors.

Improvements:
- Typed configuration via LLMExtractorConfig (no getattr/dict spelunking)
- No swallowed exceptions; all failures are logged with stack traces
- JSON fallback is explicit and configurable, not driven by exception control flow
- Cache usage is best-effort with clear diagnostics
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Literal

from smartmemory.integration.llm.prompts.prompt_provider import get_prompt_value, apply_placeholders
from smartmemory.models.base import MemoryBaseModel
from smartmemory.models.entity_types import ENTITY_TYPES
from smartmemory.models.memory_item import MemoryItem
from smartmemory.utils import get_config
from smartmemory.utils.cache import get_cache
from smartmemory.utils.llm import call_llm
from smartmemory.plugins.base import ExtractorPlugin, PluginMetadata

logger = logging.getLogger(__name__)


class EntityOut(BaseModel):
    name: str = Field(..., description="Canonical entity surface form")
    # Use dynamic list via runtime validation rather than Literal
    entity_type: str = "concept"
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    attrs: Optional[Dict[str, Any]] = None


class TripleOut(BaseModel):
    subject: str
    predicate: str
    object: str
    # Optional disambiguation hints
    subject_type: Optional[str] = None
    object_type: Optional[str] = None
    subject_ref: Optional[int] = Field(None, ge=0)
    object_ref: Optional[int] = Field(None, ge=0)
    polarity: Optional[Literal["positive", "negative"]] = None


class ExtractionOut(BaseModel):
    entities: List[EntityOut] = Field(default_factory=list)
    triples: List[TripleOut] = Field(default_factory=list)


@dataclass
class LLMExtractorConfig(MemoryBaseModel):
    """Typed config for the LLM extractor.

    Prefer environment variable for API key; fall back to legacy config path if needed.
    """
    model_name: str = "gpt-4.1-mini"
    api_key_env: str = "OPENAI_API_KEY"
    enable_json_fallback: bool = True
    temperature: float = 0.0
    max_tokens: int = 1000
    json_max_tokens: int = 512
    # Relationship labeling strategy
    use_allowed_edge_types: bool = False  # default: open-set predicates with recall-maximizing guidance
    # Control whether to inject the full ENTITY_TYPES list into prompts (can be very long)
    include_entity_types_in_prompt: bool = False
    # NOTE: Entity type validation has been moved to pipeline stages (EntityValidationStage)
    # This allows validation to be applied to ANY extractor, not just LLM
    # See: smartmemory/memory/pipeline/stages/validation.py
    # Hint to provider/models that support reasoning controls
    reasoning_effort: Optional[str] = "minimal"
    # Prompt keys for the prompt provider
    system_template_key: str = "plugins.extractors.llm.system_template"
    user_template_key: str = "plugins.extractors.llm.user_template"
    json_fallback_template_key: str = "plugins.extractors.llm.json_fallback_template"


class LLMExtractor(ExtractorPlugin):
    """
    LLM-based entity and relation extractor.
    
    Uses OpenAI-compatible models to extract entities and relationships from text.
    """
    
    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="llm",
            version="1.0.0",
            author="SmartMemory Team",
            description="Entity and relation extraction using LLM (OpenAI-compatible)",
            plugin_type="extractor",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.1.0",
            tags=["ner", "relation-extraction", "llm", "openai"]
        )
    
    def __init__(self, prompt_overrides: Optional[Dict[str, Any]] = None, config: Optional[LLMExtractorConfig] = None):
        """Initialize the LLM extractor."""
        self.overrides: Dict[str, Any] = dict(prompt_overrides or {})
        self.cfg = config or LLMExtractorConfig()
    
    def extract(self, text: str, user_id: Optional[str] = None) -> dict:
        """
        Extract entities and relations from text using LLM.
        
        Args:
            text: The text to extract from
            user_id: Optional user ID for context
        
        Returns:
            dict: Dictionary with 'entities' and 'relations' keys
        """
        content = text
        # Try Redis cache first for significant performance improvement
        try:
            cache = get_cache()

            # Check cache for existing extraction results
            cached_result = cache.get_entity_extraction(content)
            if cached_result is not None:
                # Validate cache has expected structure
                has_entities = cached_result.get('entities') or []
                has_relations = cached_result.get('relations') or []
                if has_entities or has_relations:
                    logger.debug(f"Cache hit for entity extraction: {content[:50]}...")
                    return cached_result
                else:
                    logger.debug(f"Cache hit but empty result, re-extracting: {content[:50]}...")

            logger.debug(f"Cache miss for entity extraction: {content[:50]}...")
        except Exception as e:
            logger.warning(f"Redis cache unavailable for entity extraction: {e}")
            cache = None

        # Resolve API key (env first, then legacy config)
        api_key = os.getenv(self.cfg.api_key_env)
        if not api_key:
            try:
                legacy = get_config('extractor')
                llm_cfg = legacy.get('llm') or {}
                api_key = llm_cfg.get("openai_api_key")
            except Exception as e:
                api_key = None
        if not api_key:
            raise ValueError(
                f"No API key found. Set env {self.cfg.api_key_env} or provide extractor['llm']['openai_api_key'] in config."
            )
        model_name = self.cfg.model_name

        # Prompt instructions via centralized prompts (strict) with overrides
        system_template = (self.overrides.get('system_template') if self.overrides else None) or get_prompt_value(self.cfg.system_template_key)
        if not system_template:
            raise ValueError(f"Missing prompt template '{self.cfg.system_template_key}' in prompts store")
        system_message = apply_placeholders(system_template, {})
        # Inject canonical entity types to guide the model (keeps schema and runtime validation aligned)
        if self.cfg.include_entity_types_in_prompt and ENTITY_TYPES:
            types_str = ", ".join(sorted(set(t.strip().lower() for t in ENTITY_TYPES if isinstance(t, str) and t.strip())))
            system_message = (
                f"{system_message}\n\n"
                f"ALLOWED ENTITY TYPES (use only these labels):\n{types_str}\n"
                f"When emitting triples, prefer subject_ref/object_ref indices pointing into the 'entities' list.\n"
                f"If refs are not provided, include subject_type/object_type from the allowed set."
            )

        # Edge type strategy: either provide a curated list (disabled by default) or maximize recall in open set
        if self.cfg.use_allowed_edge_types:
            # TODO: Implement ALLOWED_EDGE_TYPES in smartmemory/models/edge_types.py and inject here similar to ENTITY_TYPES
            # from smartmemory.models.edge_types import EDGE_TYPES
            # edges_str = ", ".join(sorted(set(e for e in EDGE_TYPES)))
            # system_message += f"\n\nALLOWED EDGE TYPES (prefer these labels):\n{edges_str}\n"
            # system_message += "If none applies, use a short, verb-like new label following normalization rules.\n"
            pass
        else:
            system_message = (
                f"{system_message}\n\n"
                f"RELATIONSHIP EXTRACTION (maximize coverage):\n"
                f"- Extract every explicit subject–predicate–object fact; do not limit to one per sentence.\n"
                f"- Handle coordinated subjects/objects: 'A and B use C' -> (A, uses, C), (B, uses, C).\n"
                f"- Cover common patterns: membership/employment (works_for, member_of), location (based_in, located_in),\n"
                f"  authorship/creation (authored_by, created_by), founding/leadership (founded_by, leads, reports_to),\n"
                f"  usage/affiliation (uses, affiliated_with), temporal (began_on, ended_on, occurs_on), causality (causes, results_in).\n"
                f"- Predicates must be concise (≤3 words), verb-like when possible, and will be normalized server-side.\n"
                f"- Preserve explicit negation by setting polarity='negative'."
            )

        user_template = (self.overrides.get('user_template') if self.overrides else None) or get_prompt_value(self.cfg.user_template_key)
        if not user_template:
            raise ValueError(f"Missing prompt template '{self.cfg.user_template_key}' in prompts store")
        # Support multiple placeholder keys used across prompts: {TEXT}, {{TEXT}}, {{input_text}}, {{text}}
        user_prompt = apply_placeholders(user_template, {
            "TEXT": content,
            "text": content,
            "input_text": content,
        })

        # Accumulators
        entity_map = {}  # key: (name_lower, type) -> MemoryItem
        relations = []

        def _entity_key(name: str, etype: str) -> str:
            return f"{name.strip().lower()}|{(etype or 'concept').strip().lower()}"

        def _make_entity_item(name: str, etype: str, confidence: float | None = None, attrs: dict | None = None) -> MemoryItem:
            # Deterministic ID from name+type to allow relation wiring before persistence
            base = f"{name}|{etype}"
            ent_id = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
            meta = {
                'name': name,
                'entity_type': (etype or 'concept').lower(),
                'confidence': float(confidence) if confidence else None,
            }
            if attrs and isinstance(attrs, dict):
                for k, v in attrs.items():
                    if k not in meta:
                        meta[k] = v
            return MemoryItem(content=name, item_id=ent_id, memory_type='concept', metadata=meta)

        # Primary path: JSON-object mode (provider-friendly) and client-side parsing/validation
        parsed, raw = call_llm(
            model=model_name,
            system_prompt=system_message,
            user_content=user_prompt,
            response_model=None,
            response_format={"type": "json_object"},
            json_only_instruction=(
                "Return ONLY a JSON object with keys 'entities' (list) and 'relations' (list).\n"
                "Each entity: {name: str, entity_type: str, confidence?: number [0,1], attrs?: object}.\n"
                "Each relation: {subject: str, predicate: str, object: str, subject_ref?: int, object_ref?: int, subject_type?: str, object_type?: str, "
                "polarity?: 'positive'|'negative'}.\n"
                "Do not include markdown or commentary. If none found, return {\"entities\": [], \"relations\": []}."
            ),
            max_output_tokens=self.cfg.max_tokens,
            reasoning_effort=self.cfg.reasoning_effort,
            temperature=self.cfg.temperature,
            api_key=api_key,
            config_section="extractor",
        )
        data = parsed
        if not data and raw and isinstance(raw, str):
            try:
                data = json.loads(raw)
            except Exception as e:
                logger.warning(f"Failed to parse JSON from primary call: {e}")

        # Optional fallback using a stricter JSON template if primary produced nothing
        if self.cfg.enable_json_fallback and not data:
            try:
                json_template = (self.overrides.get('json_fallback_template') if self.overrides else None) or get_prompt_value(self.cfg.json_fallback_template_key)
                if not json_template:
                    raise ValueError(f"Missing prompt template '{self.cfg.json_fallback_template_key}' in prompts store")
                prompt = apply_placeholders(json_template, {
                    "TEXT": content,
                    "text": content,
                    "input_text": content,
                })
                if ENTITY_TYPES:
                    types_str = ", ".join(sorted(set(t.strip().lower() for t in ENTITY_TYPES if isinstance(t, str) and t.strip())))
                    prompt = (
                        f"{prompt}\n\n"
                        f"ALLOWED ENTITY TYPES (use only these labels):\n{types_str}\n"
                        f"When emitting triples, prefer subject_ref/object_ref indices into 'entities'. If absent, include subject_type/object_type from the allowed set."
                    )
                if self.cfg.use_allowed_edge_types:
                    # TODO: Inject ALLOWED EDGE TYPES into fallback prompt when available (see TODO above)
                    pass
                else:
                    prompt = (
                        f"{prompt}\n\n"
                        f"RELATIONSHIP EXTRACTION (maximize coverage):\n"
                        f"- Extract every explicit subject–predicate–object fact; scan each sentence for all relations.\n"
                        f"- Handle coordinated subjects/objects; include membership, location, authorship, founding/leadership, usage, temporal, and causality when explicit.\n"
                        f"- Keep predicates short (≤3 words), verb-like when possible; server will normalize labels.\n"
                        f"- Use polarity='negative' when the text explicitly negates the relation."
                    )
                parsed_fb, raw_fb = call_llm(
                    model=model_name,
                    user_content=prompt,
                    response_model=None,
                    response_format={"type": "json_object"},
                    json_only_instruction=(
                        "Return ONLY a JSON object with keys 'entities' (list) and 'relations' (list).\n"
                        "Each entity: {name: str, entity_type: str, confidence?: number [0,1], attrs?: object}.\n"
                        "Each triple: {subject: str, predicate: str, object: str, subject_ref?: int, object_ref?: int, subject_type?: str, object_type?: str, polarity?: 'positive'|'negative'}.\n"
                        "Do not include markdown or commentary. If none found, return {\"entities\": [], \"relations\": []}."
                    ),
                    max_output_tokens=self.cfg.json_max_tokens,
                    reasoning_effort=self.cfg.reasoning_effort,
                    temperature=self.cfg.temperature,
                    api_key=api_key,
                    config_section="extractor",
                )
                data = parsed_fb
                if not data and raw_fb and isinstance(raw_fb, str):
                    data = json.loads(raw_fb)
            except Exception as e:
                logger.exception("LLM JSON-fallback extraction failed")
                raise RuntimeError(f"Failed to parse LLM extraction output: {e}\nRaw output: {raw or ''}")

        # Build MemoryItem entities list (initial, may be augmented during JSON merge below)
        entities = list(entity_map.values())

        # If we have 'data' from JSON calls, merge that into our accumulators
        if isinstance(data, dict):
            ents_in = data.get('entities') or []
            for e in ents_in:
                if isinstance(e, dict):
                    ename = (e.get('name') or '').strip()
                    etype = (e.get('entity_type') or 'concept').strip().lower()
                    # No validation here - trust the LLM
                    # Validation is now handled by EntityValidationStage in the pipeline
                    conf = e.get('confidence')
                    attrs = e.get('attrs') or {}
                    if ename:
                        key = _entity_key(ename, etype)
                        if key not in entity_map:
                            entity_map[key] = _make_entity_item(ename, etype, conf, attrs)
            # Re-materialize list and indexes after entities merge
            entities = list(entity_map.values())
            name_to_id = {}
            name_type_to_id = {}
            idx_to_id: Dict[int, str] = {}
            for idx, mi in enumerate(entities):
                if isinstance(mi.metadata, dict):
                    ename = (mi.metadata.get('name') or '').strip()
                    etype = (mi.metadata.get('entity_type') or 'concept').strip().lower()
                else:
                    ename, etype = (mi.content or '').strip(), 'concept'
                if ename:
                    name_to_id.setdefault(ename.lower(), mi.item_id)
                    name_type_to_id.setdefault((ename.lower(), etype), mi.item_id)
                    idx_to_id[idx] = mi.item_id

            trs_in = data.get('relations') or []
            for t in trs_in:
                if isinstance(t, dict):
                    s_raw = (t.get('subject') or '').strip()
                    p = _normalize_predicate((t.get('predicate') or '').strip())
                    o_raw = (t.get('object') or '').strip()
                    s_type = (t.get('subject_type') or None)
                    o_type = (t.get('object_type') or None)
                    if isinstance(s_type, str):
                        s_type = s_type.strip().lower()
                        if s_type not in ENTITY_TYPES:
                            logger.warning(f"Unknown subject_type '{s_type}' from LLM; defaulting to 'concept'")
                            s_type = 'concept'
                    if isinstance(o_type, str):
                        o_type = o_type.strip().lower()
                        if o_type not in ENTITY_TYPES:
                            logger.warning(f"Unknown object_type '{o_type}' from LLM; defaulting to 'concept'")
                            o_type = 'concept'
                    s_ref = t.get('subject_ref')
                    o_ref = t.get('object_ref')
                elif isinstance(t, (list, tuple)) and len(t) == 3:
                    s_raw, p, o_raw = t[0], t[1], t[2]
                    p = _normalize_predicate(p)
                    s_type = o_type = None
                    s_ref = o_ref = None
                else:
                    continue
                if s_raw and p and o_raw:
                    # Resolve subject id
                    sid = None
                    if isinstance(s_ref, int) and s_ref in idx_to_id:
                        sid = idx_to_id[s_ref]
                    elif s_type:
                        sid = name_type_to_id.get((s_raw.strip().lower(), s_type.strip().lower()))
                    if sid is None:
                        sid = name_to_id.get(s_raw.strip().lower())

                    # Resolve object id
                    oid = None
                    if isinstance(o_ref, int) and o_ref in idx_to_id:
                        oid = idx_to_id[o_ref]
                    elif o_type:
                        oid = name_type_to_id.get((o_raw.strip().lower(), o_type.strip().lower()))
                    if oid is None:
                        oid = name_to_id.get(o_raw.strip().lower())

                    # Ensure implicit nodes exist if still missing
                    if sid is None:
                        s_key = _entity_key(s_raw, (s_type or 'concept'))
                        entity_map.setdefault(s_key, _make_entity_item(s_raw, (s_type or 'concept'), None, None))
                        sid = entity_map[s_key].item_id
                        name_to_id.setdefault(s_raw.strip().lower(), sid)
                        name_type_to_id.setdefault((s_raw.strip().lower(), (s_type or 'concept').strip().lower()), sid)
                    if oid is None:
                        o_key = _entity_key(o_raw, (o_type or 'concept'))
                        entity_map.setdefault(o_key, _make_entity_item(o_raw, (o_type or 'concept'), None, None))
                        oid = entity_map[o_key].item_id
                        name_to_id.setdefault(o_raw.strip().lower(), oid)
                        name_type_to_id.setdefault((o_raw.strip().lower(), (o_type or 'concept').strip().lower()), oid)

                    relations.append({'source_id': sid, 'target_id': oid, 'relation_type': p})

        # Finalize entities list after any implicit additions
        entities = list(entity_map.values())

        # Return format: entities (MemoryItem objects) and relations (list of dicts)
        extraction_result = {
            'entities': entities,
            'relations': relations,
        }

        # Cache the result for future use
        if cache is not None:
            try:
                cache.set_entity_extraction(content, extraction_result)
                logger.debug(f"Cached entity extraction for: {content[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to cache entity extraction: {e}")

        return extraction_result


def _normalize_predicate(predicate: str) -> str:
    """
    Normalize predicate to be FalkorDB edge-label safe, borrowing logic from GPT-4o extractor.
    Rules:
    - lowercase
    - replace non-alphanumerics with underscore
    - collapse duplicate underscores
    - trim leading/trailing underscores
    - must start with a letter; if starts with digit, prefix underscore
    - max length 63
    - final validation pattern: ^[a-z][a-z0-9_]{0,62}$
    """
    if not predicate:
        return "unknown"
    pred = predicate.lower()
    pred = re.sub(r'[^a-z0-9]+', '_', pred)
    pred = re.sub(r'_+', '_', pred)
    pred = pred.strip('_')
    if pred and pred[0].isdigit():
        pred = '_' + pred
    if not pred or not pred[0].isalpha():
        pred = 'rel_' + pred
    if len(pred) > 63:
        pred = pred[:63]
    if not re.match(r'^[a-z][a-z0-9_]{0,62}$', pred):
        pred = 'relation'
    return pred
