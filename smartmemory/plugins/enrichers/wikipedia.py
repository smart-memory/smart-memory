from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from smartmemory.models.base import MemoryBaseModel, StageRequest
from smartmemory.plugins.base import EnricherPlugin, PluginMetadata
from smartmemory.tools.wikipedia import wikipedia_article


@dataclass
class WikipediaEnricherConfig(MemoryBaseModel):
    language: str = 'en'


@dataclass
class WikipediaEnricherRequest(StageRequest):
    language: str = 'en'
    context: Dict[str, Any] = field(default_factory=dict)
    run_id: Optional[str] = None


class WikipediaEnricher(EnricherPlugin):
    """
    Enricher that adds Wikipedia summaries and metadata for recognized entities using the wikipedia_article tool function.
    This ensures consistent, LLM/agent-compatible enrichment and DRY logic.
    """

    @classmethod
    def metadata(cls) -> PluginMetadata:
        """Return plugin metadata for discovery."""
        return PluginMetadata(
            name="wikipedia_enricher",
            version="1.0.0",
            author="SmartMemory Team",
            description="Wikipedia-based enrichment for recognized entities",
            plugin_type="enricher",
            dependencies=["wikipedia>=1.4.0"],
            min_smartmemory_version="0.1.0"
        )

    def __init__(self, config: Optional[WikipediaEnricherConfig] = None, language: Optional[str] = None):
        # Support legacy constructor signature while enforcing typed config
        if config is None and language is not None:
            config = WikipediaEnricherConfig(language=language)
        self.config = config or WikipediaEnricherConfig()
        if not isinstance(self.config, WikipediaEnricherConfig):
            raise TypeError("WikipediaEnricher requires a typed config (WikipediaEnricherConfig)")
        self.language = self.config.language

    def enrich(self, item, node_ids=None):
        entities = node_ids.get('semantic_entities', []) if isinstance(node_ids, dict) else []
        if not entities:
            return {
                'wikipedia_data': {},
                'tags': [],
                'summary': getattr(item, 'content', str(item)).split('.')[0] + '.',
                'provenance_candidates': [],
            }
        wiki_articles = wikipedia_article(entities, language=self.language)
        wiki_data = {}
        provenance_candidates = []
        for entity, article in wiki_articles.items():
            # Check if article exists (either 'exists' is True or article has content)
            exists = article.get('exists', True) if article.get('exists') is not None else bool(article.get('summary'))
            if exists:
                wikipedia_node_id = f"wikipedia:{entity.replace(' ', '_').lower()}"
                node_properties = {
                    'entity': entity,
                    'summary': article.get('summary', '')[:300],
                    'categories': article.get('categories', []),
                    'url': article.get('url'),
                    'type': 'wikipedia_article',
                }
                if hasattr(self, 'graph') and self.graph is not None:
                    self.graph.add_node(item_id=wikipedia_node_id, properties=node_properties)
                wiki_data[entity] = {
                    'exists': True,
                    'summary': node_properties['summary'],
                    'categories': node_properties['categories'],
                    'url': node_properties['url'],
                }
                provenance_candidates.append(wikipedia_node_id)
        return {
            'wikipedia_data': wiki_data,
            'tags': entities,
            'summary': getattr(item, 'content', str(item)).split('.')[0] + '.',
            'provenance_candidates': provenance_candidates,
        }
