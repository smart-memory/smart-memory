"""
Redis Caching Service for Ontology Operations

Provides fast lookups for concept/relation labels and caches LLM inference results.
"""

import hashlib
import json
import logging
from typing import List, Dict, Any, Optional

from smartmemory.configuration import MemoryConfig
from smartmemory.utils.context import get_workspace_id

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from smartmemory.ontology.ir_models import Concept, Relation, OntologyIR

logger = logging.getLogger(__name__)


class RedisOntologyCache:
    """Redis cache for ontology operations"""

    def __init__(self, config: MemoryConfig = None):
        if config is None:
            config = MemoryConfig()

        self.config = config
        host = config.cache.redis.get("host", "localhost")
        port = config.cache.redis.get("port", 9012)
        db = config.cache.redis.get("db", 0)
        ttl = config.cache.redis.get("ttl", 3600)
        self.host = host
        self.port = port
        self.db = db
        self.ttl = ttl  # Default TTL in seconds
        self.client = None

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available, caching disabled")
            return

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis connected: {host}:{port}/{db}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def _key(self, prefix: str, *args) -> str:
        """Generate cache key, optionally namespaced by workspace_id if present in context."""
        ws = get_workspace_id()
        parts = ["ontology", prefix]
        if ws:
            parts.append(str(ws))
        parts.extend(str(arg) for arg in args)
        return ":".join(parts)

    def cache_concept_mappings(self, registry_id: str, concepts: List[Concept]):
        """Cache concept label <-> ID mappings"""
        if not self.client:
            return

        try:
            pipe = self.client.pipeline()

            for concept in concepts:
                # Label -> ID mapping
                label_key = self._key("label_to_id", registry_id, concept.label.lower())
                pipe.setex(label_key, self.ttl, concept.id)

                # ID -> concept data mapping
                id_key = self._key("id_to_concept", registry_id, concept.id)
                concept_data = {
                    "id": concept.id,
                    "label": concept.label,
                    "synonyms": concept.synonyms,
                    "status": concept.status.value,
                    "confidence": concept.confidence
                }
                pipe.setex(id_key, self.ttl, json.dumps(concept_data))

                # Synonym -> ID mappings
                for synonym in concept.synonyms:
                    synonym_key = self._key("label_to_id", registry_id, synonym.lower())
                    pipe.setex(synonym_key, self.ttl, concept.id)

            pipe.execute()
            logger.debug(f"Cached {len(concepts)} concept mappings for {registry_id}")

        except Exception as e:
            logger.error(f"Failed to cache concept mappings: {e}")

    def cache_relation_mappings(self, registry_id: str, relations: List[Relation]):
        """Cache relation label <-> ID mappings"""
        if not self.client:
            return

        try:
            pipe = self.client.pipeline()

            for relation in relations:
                # Label -> ID mapping
                label_key = self._key("rel_label_to_id", registry_id, relation.label.lower())
                pipe.setex(label_key, self.ttl, relation.id)

                # ID -> relation data mapping
                id_key = self._key("id_to_relation", registry_id, relation.id)
                relation_data = {
                    "id": relation.id,
                    "label": relation.label,
                    "aliases": relation.aliases,
                    "domain": relation.domain,
                    "range": relation.range,
                    "status": relation.status.value,
                    "confidence": relation.confidence
                }
                pipe.setex(id_key, self.ttl, json.dumps(relation_data))

                # Alias -> ID mappings
                for alias in relation.aliases:
                    alias_key = self._key("rel_label_to_id", registry_id, alias.lower())
                    pipe.setex(alias_key, self.ttl, relation.id)

            pipe.execute()
            logger.debug(f"Cached {len(relations)} relation mappings for {registry_id}")

        except Exception as e:
            logger.error(f"Failed to cache relation mappings: {e}")

    def get_concept_by_label(self, registry_id: str, label: str) -> Optional[Dict[str, Any]]:
        """Get concept by label from cache"""
        if not self.client:
            return None

        try:
            # Get concept ID from label
            label_key = self._key("label_to_id", registry_id, label.lower())
            concept_id = self.client.get(label_key)

            if not concept_id:
                return None

            # Get concept data from ID
            id_key = self._key("id_to_concept", registry_id, concept_id)
            concept_data = self.client.get(id_key)

            if concept_data:
                return json.loads(concept_data)

            return None

        except Exception as e:
            logger.error(f"Failed to get concept by label: {e}")
            return None

    def get_relation_by_label(self, registry_id: str, label: str) -> Optional[Dict[str, Any]]:
        """Get relation by label from cache"""
        if not self.client:
            return None

        try:
            # Get relation ID from label
            label_key = self._key("rel_label_to_id", registry_id, label.lower())
            relation_id = self.client.get(label_key)

            if not relation_id:
                return None

            # Get relation data from ID
            id_key = self._key("id_to_relation", registry_id, relation_id)
            relation_data = self.client.get(id_key)

            if relation_data:
                return json.loads(relation_data)

            return None

        except Exception as e:
            logger.error(f"Failed to get relation by label: {e}")
            return None

    def cache_inference_result(self, prompt_hash: str, result: Dict[str, Any], ttl: Optional[int] = None):
        """Cache LLM inference result by prompt hash"""
        if not self.client:
            return

        try:
            cache_key = self._key("inference", prompt_hash)
            cache_ttl = ttl or self.ttl * 24  # Longer TTL for expensive LLM calls

            self.client.setex(cache_key, cache_ttl, json.dumps(result))
            logger.debug(f"Cached inference result for hash {prompt_hash[:8]}...")

        except Exception as e:
            logger.error(f"Failed to cache inference result: {e}")

    def get_cached_inference(self, prompt_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM inference result"""
        if not self.client:
            return None

        try:
            cache_key = self._key("inference", prompt_hash)
            cached_result = self.client.get(cache_key)

            if cached_result:
                logger.debug(f"Cache hit for inference hash {prompt_hash[:8]}...")
                return json.loads(cached_result)

            return None

        except Exception as e:
            logger.error(f"Failed to get cached inference: {e}")
            return None

    def generate_prompt_hash(self, prompt: str, params: Dict[str, Any] = None) -> str:
        """Generate hash for prompt + parameters"""
        content = prompt
        if params:
            content += json.dumps(params, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()

    def invalidate_registry_cache(self, registry_id: str):
        """Invalidate all cache entries for a registry"""
        if not self.client:
            return

        try:
            # Find all keys for this registry
            pattern = self._key("*", registry_id, "*")
            keys = self.client.keys(pattern)

            if keys:
                self.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache entries for registry {registry_id}")

        except Exception as e:
            logger.error(f"Failed to invalidate registry cache: {e}")

    def get_cache_stats(self, registry_id: str) -> Dict[str, int]:
        """Get cache statistics for a registry"""
        if not self.client:
            return {}

        try:
            stats = {}

            # Count different types of cached items
            patterns = {
                "concepts": self._key("id_to_concept", registry_id, "*"),
                "relations": self._key("id_to_relation", registry_id, "*"),
                "label_mappings": self._key("label_to_id", registry_id, "*"),
                "relation_mappings": self._key("rel_label_to_id", registry_id, "*")
            }

            for stat_name, pattern in patterns.items():
                keys = self.client.keys(pattern)
                stats[stat_name] = len(keys)

            # Get inference cache count
            inference_keys = self.client.keys(self._key("inference", "*"))
            stats["inference_cache"] = len(inference_keys)

            return stats

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def warm_cache(self, ontology: OntologyIR):
        """Warm cache with ontology data"""
        if not self.client:
            return

        try:
            # Cache approved concepts and relations
            approved_concepts = ontology.get_approved_concepts()
            approved_relations = ontology.get_approved_relations()

            self.cache_concept_mappings(ontology.registry_id, approved_concepts)
            self.cache_relation_mappings(ontology.registry_id, approved_relations)

            logger.info(f"Warmed cache for registry {ontology.registry_id}")

        except Exception as e:
            logger.error(f"Failed to warm cache: {e}")

    def search_concepts(self, registry_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search concepts by label/synonym (simple prefix matching)"""
        if not self.client:
            return []

        try:
            # Get all concept label mappings
            pattern = self._key("label_to_id", registry_id, f"{query.lower()}*")
            matching_keys = self.client.keys(pattern)

            results = []
            for key in matching_keys[:limit]:
                concept_id = self.client.get(key)
                if concept_id:
                    id_key = self._key("id_to_concept", registry_id, concept_id)
                    concept_data = self.client.get(id_key)
                    if concept_data:
                        results.append(json.loads(concept_data))

            return results

        except Exception as e:
            logger.error(f"Failed to search concepts: {e}")
            return []

    def close(self):
        """Close Redis connection"""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self.client = None
