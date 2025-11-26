"""
LLM-based ontology-aware entity and relation extractor.
Uses prompt engineering with the rich ontology schema for intelligent classification.
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any

from smartmemory.extraction.models import OntologyExtractionResponse, GenericOntologyNode
from smartmemory.integration.llm.prompts.prompt_provider import get_prompt_value, apply_placeholders
from smartmemory.models.compat.dataclass_model import get_field_names
from smartmemory.models.ontology import (
    OntologyNode, NodeType, RelationType,
    NODE_TYPE_REGISTRY
)
from smartmemory.utils import get_config
from smartmemory.utils.cache import get_cache
from smartmemory.utils.llm import run_ontology_llm

logger = logging.getLogger(__name__)


class OntologyExtractor:
    """
    Ontology-aware LLM-based extractor that uses the rich ontology schema
    for intelligent entity classification and semantic relationship extraction.
    
    This extractor produces typed entities (OntologyNode) with semantic relationships.
    For simpler extraction without ontology, use LLMExtractor from plugins.extractors.
    """

    def __init__(self):
        self.config = get_config('extractor')
        self.llm_config = self.config.get('llm') or {}
        self.model_name = self.llm_config.get('model_name', 'gpt-5-mini')

    def _build_ontology_prompt(self, template: Optional[str] = None) -> str:
        """Build a comprehensive prompt that includes the ontology schema.

        Precedence:
        1) provided template arg
        2) centralized config key 'extractor.ontology.system_template'
        """
        # Node types with descriptions
        node_types_desc = []
        for node_type in NodeType:
            node_class = NODE_TYPE_REGISTRY.get(node_type)
            if node_class:
                # Get field names from dataclass or Pydantic via compatibility helper
                fields_list = []
                for field_name in get_field_names(node_class):
                    if field_name not in ['item_id', 'created_at', 'updated_at', 'user_id', 'confidence', 'source', 'name', 'description']:
                        fields_list.append(field_name)

                node_types_desc.append(f"- {node_type.value}: {node_class.__doc__ or 'No description'}")
                if fields_list:
                    node_types_desc.append(f"  Properties: {', '.join(fields_list)}")

        # Relationship types with descriptions
        relation_types_desc = []
        for rel_type in RelationType:
            relation_types_desc.append(f"- {rel_type.value}: {rel_type.name.replace('_', ' ').title()}")

        # Load template from arg or prompts config strictly
        tpl = template or get_prompt_value('extractor.ontology.system_template')
        if not tpl:
            raise ValueError("Missing prompt template 'extractor.ontology.system_template' in prompts.json")
        prompt = apply_placeholders(tpl, {
            'node_types_desc': "\n".join(node_types_desc),
            'relation_types_desc': "\n".join(relation_types_desc),
        })
        return prompt

    def extract_entities_and_relations(
            self,
            text: str,
            user_id: Optional[str] = None,
            prompt: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            ontology_enabled: Optional[bool] = None,
            ontology_constraints: Optional[List[str]] = None,
            system_template_override: Optional[str] = None,
            user_template_override: Optional[str] = None,
            reasoning_effort: Optional[str] = None,
            include_reasoning: Optional[bool] = None,
            max_reasoning_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Extract typed entities and semantic relations using LLM with ontology schema.
        """
        # LLM calls are centralized in utils.llm.run_ontology_llm

        # Check cache first (only when no external prompt override)
        # if not prompt:
        #     cache_result = self._check_cache(text)
        #     if cache_result:
        #         return cache_result

        # Whether to apply ontology system guidance (orthogonal to prompt override)
        # Default to True to preserve prior behavior for callers that don't pass the toggle
        use_ontology = bool(ontology_enabled) if ontology_enabled is not None else True

        # Build the ontology-aware prompt if enabled
        system_prompt = None
        if use_ontology:
            system_prompt = self._build_ontology_prompt(system_template_override)
            # Optionally bias with user-provided constraints
            if ontology_constraints:
                try:
                    focus = ", ".join([str(c) for c in ontology_constraints if c])
                    if focus:
                        system_prompt = f"{system_prompt}\n\nWhen classifying, PREFER these types when applicable: {focus}."
                except Exception:
                    # Be conservative if constraints are malformed
                    pass
        # Build user prompt with method override > config (strict)
        if prompt is None:
            user_tpl = user_template_override or get_prompt_value('extractor.ontology.user_template')
            if not user_tpl:
                raise ValueError("Missing prompt template 'extractor.ontology.user_template' in prompts.json")
            user_prompt = apply_placeholders(user_tpl, {'input_text': text})
        else:
            user_prompt = None  # will use raw 'prompt' directly below

        try:
            # Resolve models/params
            _model = (model or self.model_name)
            _ = 0.1 if temperature is None else float(temperature)  # temperature currently unused
            _max_tokens = 1500 if max_tokens is None else int(max_tokens)

            user_content = f"{str(prompt)}\n\nTEXT:\n{text}" if prompt else user_prompt

            # Centralized LLM call with structured parsing and JSON fallback
            parsed, raw_response = run_ontology_llm(
                model=_model,
                system_prompt=system_prompt,
                user_content=user_content or "",
                response_model=OntologyExtractionResponse,
                max_output_tokens=max(2000, min(4000, _max_tokens)),
                reasoning_effort=reasoning_effort or "minimal",
                json_only_instruction=None,
            )

            # Parse LLM response
            if parsed is not None:
                llm_data: Dict[str, Any] = parsed
            else:
                logger.warning("Primary LLM response was empty or unparsed; attempting fallback parse")
                logger.debug(f"LLM extraction response: {raw_response}")

                # Robust JSON parsing with safe fallbacks
                llm_data = {"entities": [], "relations": []}
                try:
                    if isinstance(raw_response, str) and raw_response.strip():
                        llm_data = json.loads(raw_response)
                    else:
                        raise ValueError("Empty or non-string response from LLM")
                except Exception as e:
                    # Attempt fallback: parse function-style DSL (add_entity/add_triple)
                    logger.warning(f"Failed JSON parse; attempting fallback function-style parse: {e}")
                    logger.error(f"Raw response: {raw_response}")
                    try:
                        llm_data = self._parse_function_style_response(raw_response or "")
                    except Exception as fe:
                        logger.error(f"Non-JSON fallback parse also failed: {fe}; returning empty results")
                        llm_data = {"entities": [], "relations": []}

            # Convert LLM output to ontology nodes
            entities = self._create_ontology_nodes(llm_data.get('entities', []), user_id)
            relations = self._process_relations(llm_data.get('relations', []), entities)

            result = {
                'entities': entities,
                'relations': relations
            }

            # Cache the result if not overridden by custom prompt
            # if not prompt:
            #     self._cache_result(text, result)

            return result

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {'entities': [], 'relations': []}

    # Convenience helpers to map results into Studio-friendly JSON structures
    def to_nodes_and_triples(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert the extractor result (entities + relations) into nodes/triples JSON.
        
        Note: This returns 'triples' for backward compatibility with Studio UI.
        Internally, the system uses 'relations'.
        
        TODO: This method is currently unused. If needed for API projections,
        migrate to service layer as a presentation concern, not core extraction logic.
        Consider deprecating or moving to a separate adapter module.
        """
        entities = result.get('entities', [])
        relations = result.get('relations', [])

        nodes = []
        for ent in entities:
            # ent is an OntologyNode (dataclass-like). Use duck-typing for portability.
            node_id = getattr(ent, 'item_id', None) or getattr(ent, 'id', None)
            name = getattr(ent, 'name', None) or getattr(ent, 'text', None)
            ntype = getattr(ent, 'dynamic_node_type', None) or getattr(ent, 'node_type', None) or getattr(ent, 'type', None)
            confidence = getattr(ent, 'confidence', 0.8)
            # Pull properties from __dict__ minus well-known fields
            props = {}
            try:
                for k, v in ent.__dict__.items():
                    if k not in {"item_id", "id", "name", "text", "dynamic_node_type", "node_type", "confidence", "user_id", "source"}:
                        props[k] = v
            except Exception:
                props = {}

            nodes.append({
                "id": node_id,
                "text": name,
                "type": (ntype or "Entity"),
                "confidence": confidence,
                "properties": props,
                "provenance": [],
            })

        # Create entity reference mapping (E1/e1 -> "Apple Inc.", etc.)
        entity_map = {}
        for i, ent in enumerate(entities):
            entity_name = getattr(ent, 'name', None) or getattr(ent, 'text', None)
            if entity_name:
                # Support both uppercase and lowercase entity references
                entity_map[f"E{i + 1}"] = entity_name
                entity_map[f"e{i + 1}"] = entity_name

        triples = []
        for rel in relations:
            subj_text = rel.get('source_text') or rel.get('source') or rel.get('head')
            obj_text = rel.get('target_text') or rel.get('target') or rel.get('tail')
            pred = rel.get('relation_type') or rel.get('type') or rel.get('relation')

            # Map entity references to actual names
            if subj_text in entity_map:
                subj_text = entity_map[subj_text]
            if obj_text in entity_map:
                obj_text = entity_map[obj_text]

            # Generate consistent triple ID
            triple_key = f"{subj_text}|{pred}|{obj_text}".lower().strip()
            triple_id = f"triple_{abs(hash(triple_key)) % 10 ** 10}"

            triples.append({
                "id": triple_id,
                "subject": subj_text,
                "predicate": pred,
                "object": obj_text,
                "confidence": rel.get('confidence', 0.7),
                "provenance": [],
            })

        return {"nodes": nodes, "triples": triples}

    def _parse_function_style_response(self, text: str) -> Dict[str, Any]:
        """Parse responses of the form:
        add_entity("Apple Inc.", "organization")
        add_triple("Apple Inc.", "is_a", "technology company")

        Returns a dict with 'entities' and 'relations keys compatible with downstream processing.
        Raises ValueError if nothing could be parsed.
        """
        entities: Dict[str, Dict[str, Any]] = {}
        relationships: List[Dict[str, Any]] = []

        # Patterns allowing single or double quotes, with possible spaces
        ent_pat = re.compile(r"^\s*add_entity\(\s*([\'\"])(.*?)\1\s*,\s*([\'\"])(.*?)\3\s*\)\s*$")
        tri_pat = re.compile(r"^\s*add_triple\(\s*([\'\"])(.*?)\1\s*,\s*([\'\"])(.*?)\3\s*,\s*([\'\"])(.*?)\5\s*\)\s*$")

        for line in (text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            m_ent = ent_pat.match(line)
            if m_ent:
                name = m_ent.group(2).strip()
                etype = m_ent.group(4).strip()
                if name:
                    key = name.lower()
                    # Use last type if duplicates appear
                    entities[key] = {
                        'name': name,
                        'type': etype if etype else 'entity',
                        'properties': {}
                    }
                continue

            m_tri = tri_pat.match(line)
            if m_tri:
                subj = m_tri.group(2).strip()
                pred = m_tri.group(4).strip()
                obj = m_tri.group(6).strip()
                if subj and pred and obj:
                    relationships.append({
                        'source': subj,
                        'type': pred,
                        'target': obj,
                        'confidence': 0.7
                    })

        if not entities and not relationships:
            raise ValueError("LLM returned invalid JSON and non-JSON fallback parse found no entities/relations")

        # Convert dict to list
        ent_list = list(entities.values())
        return {'entities': ent_list, 'relations': relationships}

    def _get_response_schema(self) -> Dict[str, Any]:
        """JSON Schema for structured outputs enforcing entities/relations shape."""
        return {
            "name": "ontology_response",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "type": {"type": "string"},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                            },
                            "required": ["name", "type"],
                        }
                    },
                    "relations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "source": {"type": "string"},
                                "type": {"type": "string"},
                                "target": {"type": "string"},
                                "confidence": {"type": "number"}
                            },
                            "required": ["source", "type", "target"],
                        }
                    }
                },
                "required": ["entities", "relations"],
            },
            "strict": False,
        }

    def _create_ontology_nodes(self, llm_entities: List[Dict], user_id: Optional[str]) -> List[OntologyNode]:
        """Convert LLM entity output to proper ontology nodes."""
        nodes = []

        for entity_data in llm_entities:
            # Fail fast - no defensive programming to hide bugs
            if not isinstance(entity_data, dict):
                raise TypeError(f"Expected dict for entity_data, got {type(entity_data)}: {entity_data}")

            name = entity_data.get('name', '').strip()
            node_type_str = entity_data.get('type', '').strip()
            raw_properties = entity_data.get('properties') or {}

            if not name:
                raise ValueError(f"Entity missing required 'name' field: {entity_data}")
            if not node_type_str:
                raise ValueError(f"Entity missing required 'type' field: {entity_data}")

            # Use node type as extracted by LLM - no artificial constraints
            # The LLM can identify any meaningful entity types in context
            node_type_value = node_type_str.lower()

            # Create flexible node with LLM-extracted properties - no rigid ontology constraints
            import uuid

            # Create node with LLM-extracted properties
            node_kwargs = {
                'item_id': str(uuid.uuid4()),
                'name': name,
                'user_id': user_id,
                'source': 'llm_ontology_extractor',
                'confidence': 0.8
            }

            # Add all LLM-extracted properties without rigid validation
            # Accept either a dict or a list of {key, value} and store in 'attributes'
            if raw_properties:
                props_dict = {}
                if isinstance(raw_properties, dict):
                    props_dict = raw_properties
                elif isinstance(raw_properties, list):
                    for item in raw_properties:
                        if isinstance(item, dict):
                            k = str(item.get('key') or '').strip()
                            if k:
                                props_dict[k] = item.get('value')
                else:
                    # Unknown format; ignore but do not fail hard
                    logger.debug(f"Ignoring unsupported properties format for entity '{name}': {type(raw_properties)}")

                # Store arbitrary properties under attributes bucket to avoid invalid kwargs
                node_kwargs['attributes'] = props_dict

            # Create the flexible node with dynamic type using module-level class
            node_kwargs['dynamic_node_type'] = node_type_value
            node = GenericOntologyNode(**node_kwargs)
            nodes.append(node)

        return nodes

    def _process_relations(self, llm_relations: List[Dict], entities: List[OntologyNode]) -> List[Dict]:
        """Process LLM relationship output and validate semantic correctness."""
        relations = []

        # Normalization helper: lowercase, replace non-alphanum with spaces, collapse spaces
        def _norm(s: str) -> str:
            s = (s or '').lower()
            s = re.sub(r'[^a-z0-9]+', ' ', s)
            return re.sub(r'\s+', ' ', s).strip()

        # Create entity lookup by normalized name
        entity_lookup = {_norm(entity.name): entity for entity in entities}

        for rel_data in llm_relations:
            # Fail fast - no defensive programming to hide bugs
            if not isinstance(rel_data, dict):
                raise TypeError(f"Expected dict for rel_data, got {type(rel_data)}: {rel_data}")

            source_name = rel_data.get('source', '').strip()
            target_name = rel_data.get('target', '').strip()
            rel_type_str = rel_data.get('type', '').strip()
            confidence = rel_data.get('confidence', 0.7)

            if not source_name:
                raise ValueError(f"Relation missing required 'source' field: {rel_data}")
            if not target_name:
                raise ValueError(f"Relation missing required 'target' field: {rel_data}")
            if not rel_type_str:
                raise ValueError(f"Relation missing required 'type' field: {rel_data}")

            # Find corresponding entities using normalized lookup; if not found, proceed without IDs
            source_entity = entity_lookup.get(_norm(source_name))
            target_entity = entity_lookup.get(_norm(target_name))

            # Use relationship type as extracted by LLM - no artificial constraints
            # The LLM knows best what relationships are meaningful in the context
            rel_type_value = rel_type_str.upper().replace(' ', '_')

            rel_out = {
                'relation_type': rel_type_value,
                'confidence': confidence,
                'source_text': source_name,
                'target_text': target_name,
            }
            if source_entity:
                rel_out['source_id'] = source_entity.item_id
            else:
                logger.debug(f"Source entity not found for relation; leaving ID empty: '{source_name}' -> '{target_name}'")
            if target_entity:
                rel_out['target_id'] = target_entity.item_id
            else:
                logger.debug(f"Target entity not found for relation; leaving ID empty: '{source_name}' -> '{target_name}'")

            relations.append(rel_out)

        return relations

    def _check_cache(self, text: str) -> Optional[Dict]:
        """Check Redis cache for existing extraction results."""
        try:
            cache = get_cache()
            cached_result = cache.get_entity_extraction(text)
            if cached_result:
                logger.debug(f"Cache hit for LLM ontology extraction: {text[:50]}...")
                return cached_result
        except Exception as e:
            logger.debug(f"Cache check failed: {e}")
        return None

    def _cache_result(self, text: str, result: Dict):
        """Cache extraction result."""
        try:
            cache = get_cache()
            cache.set_entity_extraction(text, result)
            logger.debug(f"Cached LLM ontology extraction for: {text[:50]}...")
        except Exception as e:
            logger.debug(f"Cache storage failed: {e}")


def create_ontology_aware_extractor():
    """Factory function to create the ontology-aware extractor (LLM-based)."""
    return OntologyExtractor()
