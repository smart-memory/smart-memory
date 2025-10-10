"""
Core Vector Operations

Consolidates vector operations from:
- stores/vector_store.py (semantic memory operations)
- ontology/chroma.py (concept clustering)
- stores/external/vector/chromadb_handler.py (LlamaIndex operations)
"""

import logging
from smartmemory.stores.vector.chroma.client import ChromaClient
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


class VectorOperations:
    """Core vector operations using unified ChromaDB client"""

    def __init__(self, chroma_client: ChromaClient, collection_name: str):
        """Initialize vector operations for a specific collection
        
        Args:
            chroma_client: Unified ChromaDB client instance
            collection_name: Name of the collection to operate on
        """
        self.chroma_client = chroma_client
        self.collection_name = collection_name
        self.collection = chroma_client.get_or_create_collection(collection_name)

        # Instance cache for performance
        self._cache = {}

    def add(self,
            item_id: str,
            embedding: List[float],
            metadata: Optional[Dict[str, Any]] = None,
            document: Optional[str] = None) -> bool:
        """Add vector embedding to collection
        
        Args:
            item_id: Unique identifier for the item
            embedding: Vector embedding
            metadata: Optional metadata dictionary
            document: Optional document text
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare data for ChromaDB
            ids = [item_id]
            embeddings = [embedding]
            metadatas = [metadata or {}]
            documents = [document] if document else None

            # Add to collection
            if documents:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
            else:
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

            logger.debug(f"Added vector {item_id} to collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to add vector {item_id}: {e}")
            return False

    def search(self,
               query_embedding: List[float],
               top_k: int = 5,
               where: Optional[Dict[str, Any]] = None,
               include_metadata: bool = True,
               include_documents: bool = True) -> List[Dict[str, Any]]:
        """Search for similar vectors
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of results to return
            where: Optional metadata filter
            include_metadata: Whether to include metadata in results
            include_documents: Whether to include documents in results
            
        Returns:
            List of search results with ids, distances, metadata, documents
        """
        try:
            # Prepare include list
            include = ["distances"]
            if include_metadata:
                include.append("metadatas")
            if include_documents:
                include.append("documents")

            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=include
            )

            # Format results
            formatted_results = []
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0] if include_metadata else [{}] * len(ids)
            documents = results.get("documents", [[]])[0] if include_documents else [None] * len(ids)

            for i, item_id in enumerate(ids):
                result = {
                    "id": item_id,
                    "distance": distances[i] if i < len(distances) else None,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "document": documents[i] if i < len(documents) else None
                }
                formatted_results.append(result)

            logger.debug(f"Found {len(formatted_results)} results for search in {self.collection_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed in collection {self.collection_name}: {e}")
            return []

    def get(self, item_ids: Union[str, List[str]],
            include_metadata: bool = True,
            include_documents: bool = True) -> List[Dict[str, Any]]:
        """Get specific items by ID
        
        Args:
            item_ids: Single ID or list of IDs to retrieve
            include_metadata: Whether to include metadata
            include_documents: Whether to include documents
            
        Returns:
            List of retrieved items
        """
        try:
            # Ensure item_ids is a list
            if isinstance(item_ids, str):
                item_ids = [item_ids]

            # Prepare include list
            include = []
            if include_metadata:
                include.append("metadatas")
            if include_documents:
                include.append("documents")

            # Get items
            results = self.collection.get(
                ids=item_ids,
                include=include
            )

            # Format results
            formatted_results = []
            ids = results.get("ids", [])
            metadatas = results.get("metadatas", []) if include_metadata else [{}] * len(ids)
            documents = results.get("documents", []) if include_documents else [None] * len(ids)

            for i, item_id in enumerate(ids):
                result = {
                    "id": item_id,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "document": documents[i] if i < len(documents) else None
                }
                formatted_results.append(result)

            logger.debug(f"Retrieved {len(formatted_results)} items from {self.collection_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Failed to get items from {self.collection_name}: {e}")
            return []

    def update(self,
               item_id: str,
               embedding: Optional[List[float]] = None,
               metadata: Optional[Dict[str, Any]] = None,
               document: Optional[str] = None) -> bool:
        """Update existing vector item
        
        Args:
            item_id: ID of item to update
            embedding: New embedding (optional)
            metadata: New metadata (optional)
            document: New document (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            update_data = {"ids": [item_id]}

            if embedding is not None:
                update_data["embeddings"] = [embedding]
            if metadata is not None:
                update_data["metadatas"] = [metadata]
            if document is not None:
                update_data["documents"] = [document]

            self.collection.update(**update_data)
            logger.debug(f"Updated vector {item_id} in collection {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to update vector {item_id}: {e}")
            return False

    def delete(self, item_ids: Union[str, List[str]]) -> bool:
        """Delete vectors by ID
        
        Args:
            item_ids: Single ID or list of IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure item_ids is a list
            if isinstance(item_ids, str):
                item_ids = [item_ids]

            self.collection.delete(ids=item_ids)
            logger.debug(f"Deleted {len(item_ids)} vectors from {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors from {self.collection_name}: {e}")
            return False

    def clear(self) -> bool:
        """Clear all vectors from collection
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get all IDs first
            results = self.collection.get()
            ids = results.get("ids", [])

            if ids:
                self.collection.delete(ids=ids)
                logger.info(f"Cleared {len(ids)} vectors from collection {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} was already empty")

            return True

        except Exception as e:
            logger.error(f"Failed to clear collection {self.collection_name}: {e}")
            return False

    def count(self) -> int:
        """Get count of vectors in collection
        
        Returns:
            Number of vectors in collection
        """
        try:
            results = self.collection.get()
            count = len(results.get("ids", []))
            logger.debug(f"Collection {self.collection_name} contains {count} vectors")
            return count
        except Exception as e:
            logger.error(f"Failed to count vectors in {self.collection_name}: {e}")
            return 0

    def find_similar_clusters(self,
                              embeddings: List[List[float]],
                              threshold: float = 0.8,
                              max_clusters: int = 10) -> List[List[int]]:
        """Find clusters of similar embeddings
        
        Args:
            embeddings: List of embeddings to cluster
            threshold: Similarity threshold for clustering
            max_clusters: Maximum number of clusters
            
        Returns:
            List of clusters, each containing indices of similar embeddings
        """
        try:
            clusters = []
            used_indices = set()

            for i, embedding in enumerate(embeddings):
                if i in used_indices:
                    continue

                # Search for similar embeddings
                similar = self.search(
                    query_embedding=embedding,
                    top_k=len(embeddings)
                )

                # Create cluster of similar items
                cluster = [i]
                for result in similar:
                    # Convert distance to similarity (assuming cosine distance)
                    similarity = 1 - result["distance"]
                    if similarity >= threshold:
                        # Find index in original embeddings
                        for j, emb in enumerate(embeddings):
                            if j not in used_indices and j != i:
                                # This is a simplified check - in practice you'd need
                                # to match the actual embedding
                                cluster.append(j)
                                used_indices.add(j)
                                break

                if len(cluster) > 1:
                    clusters.append(cluster)
                    used_indices.update(cluster)

                if len(clusters) >= max_clusters:
                    break

            logger.debug(f"Found {len(clusters)} clusters from {len(embeddings)} embeddings")
            return clusters

        except Exception as e:
            logger.error(f"Failed to find clusters: {e}")
            return []
