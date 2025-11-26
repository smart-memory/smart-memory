import json
from typing import Optional, List, Dict, Any, Set, Tuple

from smartmemory.memory.pipeline.stages.clustering import logger


class LLMClustering:
    """
    LLM-based semantic clustering for entities and relations.

    This uses an LLM to identify semantic equivalence:
    - Entity aliases: "Joe" ↔ "Joseph", "ML" ↔ "machine learning"
    - Relation synonyms: "is type of" ↔ "is a kind of"

    This is more powerful than embedding-based clustering for catching
    semantic equivalence that embeddings might miss.
    """

    def __init__(
        self,
        model_name: str = "gpt-5-mini",
        context: Optional[str] = None
    ):
        """
        Initialize LLM clustering.

        Args:
            model_name: LLM model to use for clustering
            context: Optional domain context to guide clustering
                     e.g., "Family relationships", "Machine learning concepts"
        """
        self.model_name = model_name
        self.context = context

    def cluster_entities(
        self,
        entities: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> Dict[str, Set[str]]:
        """
        Cluster entities by semantic equivalence using LLM.

        Args:
            entities: List of entity dicts with 'name' and optionally 'type'
            context: Optional context to guide clustering

        Returns:
            Dict mapping canonical name to set of equivalent names
            e.g., {'Joseph': {'Joe', 'Joseph', 'Joey'}}
        """
        if not entities:
            return {}

        # Extract entity names
        entity_names = []
        for e in entities:
            name = e.get('name') or e.get('content', '')
            if name:
                entity_names.append(name)

        if len(entity_names) < 2:
            return {}

        # Use LLM to find clusters
        clusters = self._llm_cluster_names(entity_names, context or self.context, "entities")

        return clusters

    def cluster_relations(
        self,
        relations: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Set[str]]:
        """
        Cluster relation predicates by semantic equivalence.

        Args:
            relations: List of relation predicate strings
            context: Optional context to guide clustering

        Returns:
            Dict mapping canonical predicate to set of equivalent predicates
            e.g., {'is_type_of': {'is type of', 'is a kind of', 'is a type of'}}
        """
        if not relations or len(relations) < 2:
            return {}

        # Deduplicate first
        unique_relations = list(set(relations))

        if len(unique_relations) < 2:
            return {}

        clusters = self._llm_cluster_names(unique_relations, context or self.context, "relations")

        return clusters

    def _llm_cluster_names(
        self,
        names: List[str],
        context: Optional[str],
        item_type: str  # "entities" or "relations"
    ) -> Dict[str, Set[str]]:
        """
        Use LLM to cluster names by semantic equivalence.

        Args:
            names: List of names to cluster
            context: Optional domain context
            item_type: Type of items being clustered

        Returns:
            Dict mapping canonical name to set of equivalent names
        """
        from smartmemory.utils.llm import call_llm

        # Build prompt
        context_str = f"\nContext: {context}" if context else ""

        system_prompt = f"""You are clustering {item_type} by semantic equivalence.
Group together {item_type} that refer to the same thing, even if spelled differently.

Examples of equivalent {item_type}:
- Names: "Joe", "Joseph", "Joey" → same person
- Abbreviations: "ML", "machine learning" → same concept
- Variations: "is type of", "is a kind of" → same relation

Return a JSON object with a 'clusters' array.
Each cluster should have:
- 'canonical': The most complete/formal name to use
- 'members': Array of all equivalent names (including canonical)

Only create clusters for {item_type} that are actually equivalent.
{item_type.capitalize()} that are unique should NOT appear in any cluster."""

        user_prompt = f"""{item_type.upper()} TO CLUSTER:{context_str}
{json.dumps(names, indent=2)}

Find groups of semantically equivalent {item_type}. Return JSON with 'clusters' array."""

        try:
            parsed, raw = call_llm(
                model=self.model_name,
                system_prompt=system_prompt,
                user_content=user_prompt,
                response_format={"type": "json_object"},
                json_only_instruction="Return ONLY JSON with 'clusters' array.",
                max_output_tokens=2000,
                temperature=0.0,
            )

            data = parsed or {}
            if not data and raw and isinstance(raw, str):
                try:
                    data = json.loads(raw)
                except Exception:
                    pass

            # Parse clusters
            raw_clusters = data.get('clusters', [])
            result = {}

            for cluster in raw_clusters:
                if not isinstance(cluster, dict):
                    continue
                canonical = cluster.get('canonical', '')
                members = cluster.get('members', [])

                if canonical and members and len(members) > 1:
                    result[canonical] = set(members)

            logger.info(f"LLM clustering found {len(result)} {item_type} clusters")
            return result

        except Exception as e:
            logger.error(f"LLM clustering failed: {e}")
            return {}

    def apply_entity_clusters(
        self,
        entities: List[Dict[str, Any]],
        clusters: Dict[str, Set[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Apply entity clusters to deduplicate entities.

        Args:
            entities: List of entity dicts
            clusters: Cluster mapping from cluster_entities()

        Returns:
            Tuple of (deduplicated entities, name_to_canonical mapping)
        """
        # Build reverse mapping: any name → canonical name
        name_to_canonical = {}
        for canonical, members in clusters.items():
            for member in members:
                name_to_canonical[member.lower()] = canonical

        # Deduplicate entities
        seen_canonical = set()
        deduplicated = []

        for entity in entities:
            name = entity.get('name') or entity.get('content', '')
            if not name:
                continue

            # Get canonical name
            canonical = name_to_canonical.get(name.lower(), name)

            if canonical.lower() in seen_canonical:
                continue

            seen_canonical.add(canonical.lower())

            # Update entity with canonical name
            entity_copy = dict(entity)
            if canonical != name:
                entity_copy['name'] = canonical
                entity_copy['aliases'] = entity_copy.get('aliases', [])
                if name not in entity_copy['aliases']:
                    entity_copy['aliases'].append(name)

            deduplicated.append(entity_copy)

        return deduplicated, name_to_canonical

    def apply_relation_clusters(
        self,
        relations: List[Dict[str, Any]],
        clusters: Dict[str, Set[str]]
    ) -> List[Dict[str, Any]]:
        """
        Apply relation clusters to normalize predicates.

        Args:
            relations: List of relation dicts with 'predicate' or 'relation_type'
            clusters: Cluster mapping from cluster_relations()

        Returns:
            Relations with normalized predicates
        """
        # Build reverse mapping
        pred_to_canonical = {}
        for canonical, members in clusters.items():
            for member in members:
                pred_to_canonical[member.lower()] = canonical

        normalized = []
        for rel in relations:
            rel_copy = dict(rel)

            # Try both 'predicate' and 'relation_type' keys
            for key in ['predicate', 'relation_type']:
                if key in rel_copy:
                    original = rel_copy[key]
                    canonical = pred_to_canonical.get(original.lower(), original)
                    if canonical != original:
                        rel_copy[key] = canonical
                        rel_copy['original_predicate'] = original

            normalized.append(rel_copy)

        return normalized
