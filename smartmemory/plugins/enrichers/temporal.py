import json
import logging
import openai
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from smartmemory.integration.llm.prompts.prompt_provider import get_prompt_value, apply_placeholders
from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata


@dataclass
class TemporalEnricherConfig(MemoryBaseModel):
    model_name: str = 'gpt-3.5-turbo'
    openai_api_key: Optional[str] = None
    prompt_template_key: str = 'enrichers.temporal.prompt_template'


@dataclass
class TemporalEnricherRequest(StageRequest):
    model_name: str = 'gpt-3.5-turbo'
    openai_api_key: Optional[str] = None
    prompt_template_key: str = 'enrichers.temporal.prompt_template'
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class TemporalEnricher(EnricherPlugin):
    """
    Uses OpenAI LLM to infer temporal (bitemporal) metadata for entities and relations from content/metadata.
    Adds a 'temporal' field to the enrichment result (does not intersect with other enrichers).
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="temporal_enricher",
            version="1.0.0",
            author="SmartMemory Team",
            description="LLM-based temporal metadata inference for entities and relations",
            plugin_type="enricher",
            dependencies=["openai>=1.0.0"],
            min_smartmemory_version="0.1.0"
        )

    def __init__(self, config: Optional[TemporalEnricherConfig] = None):
        self.config = config or TemporalEnricherConfig()
        if not isinstance(self.config, TemporalEnricherConfig):
            raise TypeError("TemporalEnricher requires a typed config (TemporalEnricherConfig)")
        if self.config.openai_api_key:
            openai.api_key = self.config.openai_api_key
        self.model = self.config.model_name

    def enrich(self, item, node_ids=None, prompt_template=None):
        content = getattr(item, 'content', str(item))
        entities = node_ids.get('semantic_entities', []) if isinstance(node_ids, dict) else []
        template_key = self.config.prompt_template_key
        template = prompt_template or get_prompt_value(template_key)
        if not template:
            raise ValueError(f"Missing prompt template '{template_key}' in prompts.json")
        prompt = apply_placeholders(template, {"TEXT": content, "ENTITIES": json.dumps(entities)})
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"}
            )
            result = response.choices[0].message.content
            temporal = json.loads(result)
        except Exception:
            logging.exception("TemporalEnricher: failed to obtain or parse OpenAI response")
            temporal = {}
        return {'temporal': temporal}
