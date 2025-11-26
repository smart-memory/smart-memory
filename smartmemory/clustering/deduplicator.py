from typing import Optional, Dict, Any

from smartmemory.clustering.embedding import EmbeddingClusterer
from smartmemory.clustering.llm import LLMClustering


class HybridDeduplicator:
    """
    Hybrid deduplication combining SemHash + KMeans + LLM.

    FULL deduplication method:
    1. SemHash: Fast deterministic dedup (catches obvious duplicates)
    2. KMeans: Cluster remaining items by embedding similarity
    3. LLM: Semantic dedup within each cluster (parallel)

    This provides the best balance of speed and accuracy.
    """

    def __init__(
        self,
        semhash_threshold: float = 0.95,
        embedding_model: str = "all-MiniLM-L6-v2",
        cluster_size: int = 128,
        llm_model: str = "gpt-5-mini",
        context: Optional[str] = None
    ):
        """
        Initialize hybrid deduplicator.

        Args:
            semhash_threshold: SemHash similarity threshold
            embedding_model: Model for embeddings
            cluster_size: Target cluster size for KMeans
            llm_model: LLM model for semantic dedup
            context: Optional domain context
        """
        self.semhash_threshold = semhash_threshold
        self.embedding_model = embedding_model
        self.cluster_size = cluster_size
        self.llm_model = llm_model
        self.context = context

    def deduplicate(
        self,
        extraction_result: Dict[str, Any],
        method: str = "full"
    ) -> Dict[str, Any]:
        """
        Deduplicate extraction result using hybrid approach.

        Args:
            extraction_result: Dict with 'entities' and 'relations'
            method: Deduplication method
                - "semhash": SemHash only (fast)
                - "kmeans_llm": KMeans + LLM (no SemHash)
                - "full": SemHash + KMeans + LLM (best quality)

        Returns:
            Deduplicated extraction result with cluster mappings
        """
        from smartmemory.utils.deduplication import (
            semhash_deduplicate_entities,
            semhash_deduplicate_relations
        )

        entities = extraction_result.get('entities', [])
        relations = extraction_result.get('relations', [])

        result = dict(extraction_result)
        entity_clusters = {}
        edge_clusters = {}

        # Step 1: SemHash deduplication (if not kmeans_llm only)
        if method in ["semhash", "full"]:
            if entities:
                entities, semhash_entity_clusters = semhash_deduplicate_entities(
                    entities, self.semhash_threshold
                )
                entity_clusters.update({k: set(v) for k, v in semhash_entity_clusters.items()})

            if relations:
                predicates = [r.get('predicate') or r.get('relation_type', '') for r in relations]
                _, semhash_edge_clusters = semhash_deduplicate_relations(
                    predicates, self.semhash_threshold
                )
                edge_clusters.update({k: set(v) for k, v in semhash_edge_clusters.items()})

        # Step 2: KMeans + LLM deduplication (if not semhash only)
        if method in ["kmeans_llm", "full"] and len(entities) >= 2:
            clusterer = EmbeddingClusterer(
                embedding_model=self.embedding_model,
                cluster_size=self.cluster_size
            )
            llm_clusterer = LLMClustering(
                model_name=self.llm_model,
                context=self.context
            )

            # Cluster entities
            entity_names = [e.get('name') or e.get('content', '') for e in entities if e]
            item_clusters = clusterer.cluster_items(entity_names, "entity")

            # Run LLM dedup on each cluster (could be parallelized)
            for cluster in item_clusters:
                if len(cluster) < 2:
                    continue

                cluster_entities = [{'name': name} for name in cluster]
                llm_clusters = llm_clusterer.cluster_entities(cluster_entities, self.context)

                for canonical, members in llm_clusters.items():
                    if canonical not in entity_clusters:
                        entity_clusters[canonical] = set()
                    entity_clusters[canonical].update(members)

        # Apply clusters to entities
        if entity_clusters:
            # Build name -> canonical mapping
            name_to_canonical = {}
            for canonical, members in entity_clusters.items():
                for member in members:
                    name_to_canonical[member.lower()] = canonical

            # Deduplicate entities
            seen = set()
            deduped_entities = []
            for entity in entities:
                name = entity.get('name') or entity.get('content', '')
                canonical = name_to_canonical.get(name.lower(), name)
                if canonical.lower() not in seen:
                    seen.add(canonical.lower())
                    entity_copy = dict(entity)
                    if canonical != name:
                        entity_copy['name'] = canonical
                        entity_copy['aliases'] = entity_copy.get('aliases', [])
                        if name not in entity_copy['aliases']:
                            entity_copy['aliases'].append(name)
                    deduped_entities.append(entity_copy)

            result['entities'] = deduped_entities

        # Store cluster mappings
        if entity_clusters:
            result['entity_clusters'] = {k: list(v) for k, v in entity_clusters.items()}
        if edge_clusters:
            result['edge_clusters'] = {k: list(v) for k, v in edge_clusters.items()}

        return result
