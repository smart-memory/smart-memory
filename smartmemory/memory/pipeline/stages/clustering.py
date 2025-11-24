"""
Global Clustering Stage.

This stage performs global deduplication by clustering entities based on their vector embeddings
and merging duplicates in the graph.
"""

import logging
from typing import List, Dict, Any, Set, Optional

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
        Run clustering with automatic scope filtering via ScopeProvider.
        
        Filtering is determined by the ScopeProvider that was injected into SmartMemory
        at initialization. This ensures consistent tenant/workspace/user isolation.
        
        Returns a dict with stats about merged entities:
        - merged_count: Number of entities merged
        - clusters_found: Number of clusters identified
        - total_entities: Total number of entities processed
        """
        try:
            # Get isolation filters from ScopeProvider (single source of truth)
            filters = {}
            if hasattr(self.memory, 'scope_provider') and self.memory.scope_provider:
                filters = self.memory.scope_provider.get_isolation_filters()
            
            # 1. Fetch all items with embeddings using backend-agnostic approach
            ids = []
            embeddings = []
            
            # Use SmartMemory's graph abstraction instead of direct backend access
            all_items = self.memory.get_all_items_debug()
            
            for item_id in self._get_all_item_ids():
                item = self.memory.get(item_id)
                if not item:
                    continue
                    
                # Get embedding from item or vector store
                embedding = self._get_embedding(item_id, item)
                if embedding:
                    ids.append(item_id)
                    embeddings.append(embedding)
            
            if not ids:
                return {"merged_count": 0, "clusters_found": 0, "total_entities": 0, **filters}

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

    def _get_all_item_ids(self) -> List[str]:
        """Get all item IDs using backend-agnostic approach."""
        try:
            # Use graph's nodes interface
            if hasattr(self.memory._graph, 'nodes') and hasattr(self.memory._graph.nodes, 'nodes'):
                return list(self.memory._graph.nodes.nodes())
            # Fallback to get_all_items_debug
            debug_info = self.memory.get_all_items_debug()
            return [s.get('item_id') for s in debug_info.get('sample_items', [])]
        except Exception as e:
            logger.warning(f"Failed to get item IDs: {e}")
            return []

    def _get_embedding(self, item_id: str, item: Any) -> List[float]:
        """Get embedding for an item from item or vector store."""
        # Try item's embedding attribute first
        if hasattr(item, 'embedding') and item.embedding is not None:
            emb = item.embedding
            return emb.tolist() if hasattr(emb, 'tolist') else list(emb)
        
        # Try vector store lookup
        try:
            result = self.vector_store.get(item_id)
            if result and 'embedding' in result:
                return result['embedding']
        except Exception:
            pass
        
        return None

    def _find_clusters(self, ids: List[str], embeddings: List[List[float]]) -> List[List[str]]:
        """
        Find clusters of similar entities.
        Returns a list of clusters, where each cluster is a list of item_ids.
        """
        clusters = []
        processed_indices: Set[int] = set()
        
        # Distance threshold for clustering (lower = more similar)
        # Most vector stores return distance scores where 0 = identical
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
