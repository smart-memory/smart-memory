"""
Global Clustering Stage.

This stage performs global deduplication by clustering entities based on their vector embeddings
and merging duplicates in the graph.
"""

import logging
from typing import List, Dict, Any, Set

from smartmemory.stores.vector.vector_store import VectorStore
from smartmemory.models.memory_item import MemoryItem

logger = logging.getLogger(__name__)


class GlobalClustering:
    """
    Pipeline stage that clusters entities across the entire graph and merges duplicates.
    This is typically run as a background maintenance task, not on every ingestion.
    """

    def __init__(self, memory_instance):
        self.memory = memory_instance
        self.vector_store = VectorStore()

    def run(self) -> Dict[str, Any]:
        """
        Run global clustering.
        
        Returns a dict with stats about merged entities:
        - merged_count: Number of entities merged
        - clusters_found: Number of clusters identified
        - total_entities: Total number of entities processed
        """
        try:
            # 1. Fetch all embeddings
            # Access FalkorDB backend directly
            if not hasattr(self.vector_store, '_backend') or not hasattr(self.vector_store._backend, 'label'):
                logger.warning("Vector store backend is not FalkorDB or incompatible. Skipping clustering.")
                return {"skipped": True, "reason": "backend_incompatible"}
            
            backend = self.vector_store._backend
            label = backend.label
            
            # Fetch all nodes with embeddings
            # Note: Fetching all embeddings might be heavy. 
            # In a production system, we might want to paginate or process in chunks.
            query = f"MATCH (n:{label}) RETURN n.id as id, n.embedding as embedding"
            res = backend.graph.query(query)
            
            ids = []
            embeddings = []
            
            if hasattr(res, 'result_set'):
                for row in res.result_set:
                    if len(row) >= 2:
                        ids.append(row[0])
                        embeddings.append(row[1])
            
            if not ids:
                return {"merged_count": 0, "clusters_found": 0, "total_entities": 0}

            # 2. Cluster embeddings
            # We use a simple greedy clustering approach leveraging the vector index
            clusters = self._find_clusters(ids, embeddings)
            
            # 3. Merge clusters
            merged_count = 0
            for cluster in clusters:
                if len(cluster) < 2:
                    continue
                    
                # Identify canonical entity (e.g., longest name or specific criteria)
                canonical_id = self._identify_canonical(cluster)
                source_ids = [pid for pid in cluster if pid != canonical_id]
                
                if not source_ids:
                    continue
                    
                # Merge in graph
                success = self.memory._graph.backend.merge_nodes(canonical_id, source_ids)
                if success:
                    merged_count += len(source_ids)
                    
                    # Also remove merged vectors from vector store
                    self.vector_store.delete(source_ids)

            return {
                "merged_count": merged_count, 
                "clusters_found": len(clusters),
                "total_entities": len(ids)
            }

        except Exception as e:
            logger.error(f"Global clustering failed: {e}")
            return {"error": str(e)}

    def _find_clusters(self, ids: List[str], embeddings: List[List[float]]) -> List[List[str]]:
        """
        Find clusters of similar entities.
        Returns a list of clusters, where each cluster is a list of item_ids.
        """
        clusters = []
        processed_indices: Set[int] = set()
        
        # Threshold for similarity. 
        # FalkorDB vector search returns 'score'. 
        # If using Euclidean distance, lower is better.
        # If using Cosine Similarity, higher is better (usually).
        # FalkorDB documentation says: "The score is the distance between the query vector and the node vector."
        # So lower is better.
        # For cosine distance, range is [0, 2]. 0 is identical.
        # Let's assume a strict threshold.
        distance_threshold = 0.1 

        for i, embedding in enumerate(embeddings):
            if i in processed_indices:
                continue
                
            current_id = ids[i]
            
            # Search for similar items using the vector store
            # We use the embedding directly
            results = self.vector_store.search(embedding, top_k=10, is_global=True)
            
            cluster = [current_id]
            processed_indices.add(i)
            
            for result in results:
                res_id = result['id']
                score = result.get('score', 1.0) 
                
                # Check if result is the item itself
                if res_id == current_id:
                    continue
                    
                # Find index of res_id in our local list to mark as processed
                try:
                    res_idx = ids.index(res_id)
                except ValueError:
                    continue 
                    
                if res_idx in processed_indices:
                    continue

                # Check similarity
                if score < distance_threshold: # Very close
                     cluster.append(res_id)
                     processed_indices.add(res_idx)

            if len(cluster) > 1:
                clusters.append(cluster)
                
        return clusters

    def _identify_canonical(self, cluster_ids: List[str]) -> str:
        """
        Identify the canonical entity ID from a cluster.
        Strategy: Prefer the one with the longest name (most descriptive), 
        or highest confidence if available.
        """
        # Fetch full nodes to check properties
        candidates = []
        for item_id in cluster_ids:
            node = self.memory.get(item_id)
            if node:
                candidates.append(node)
        
        if not candidates:
            return cluster_ids[0]
            
        # Sort by name length (descending) as a proxy for completeness
        # Also consider 'confidence' metadata if present
        def score_candidate(item):
            name = ""
            if isinstance(item, MemoryItem):
                name = item.metadata.get('name', item.content)
            elif isinstance(item, dict):
                name = item.get('name', item.get('content', ''))
            else:
                name = str(item)
                
            return len(name)

        candidates.sort(key=score_candidate, reverse=True)
        
        best = candidates[0]
        if isinstance(best, MemoryItem):
            return best.item_id
        elif isinstance(best, dict):
            return best.get('item_id')
        else:
            return cluster_ids[0]
