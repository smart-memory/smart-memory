"""
Ontology Vector Service

Consolidates ontology-specific vector operations from:
- ontology/chroma.py (semantic clustering for concepts/relations)
"""

import logging
from smartmemory.stores.vector.chroma.client import ChromaClient
from typing import Dict, List, Optional, Any

from smartmemory.stores.vector.vector_operations import VectorOperations

logger = logging.getLogger(__name__)


class OntologyVectorService:
    """Ontology-specific vector operations for semantic clustering"""

    def __init__(self, config=None):
        """Initialize ontology vector service
        
        Args:
            config: MemoryConfig object (optional)
        """
        self.chroma_client = ChromaClient(config)

        # Initialize collections for different ontology types
        self.candidate_terms = VectorOperations(
            self.chroma_client,
            "candidate_terms"
        )
        self.ontology_labels = VectorOperations(
            self.chroma_client,
            "ontology_labels"
        )
        self.external_ontology_labels = VectorOperations(
            self.chroma_client,
            "external_ontology_labels"
        )

        logger.info("Ontology vector service initialized")

    def add_candidate_term(self,
                           term: str,
                           embedding: List[float],
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add candidate term for clustering
        
        Args:
            term: The candidate term text
            embedding: Vector embedding for the term
            metadata: Optional metadata (frequency, context, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        term_id = self._generate_term_id(term)

        # Add term metadata
        term_metadata = {
            "term": term,
            "type": "candidate",
            **(metadata or {})
        }

        return self.candidate_terms.add(
            item_id=term_id,
            embedding=embedding,
            metadata=term_metadata,
            document=term
        )

    def add_ontology_label(self,
                           label: str,
                           embedding: List[float],
                           concept_type: str,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add ontology label for similarity matching
        
        Args:
            label: The ontology label text
            embedding: Vector embedding for the label
            concept_type: Type of concept (entity, relation, etc.)
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        label_id = self._generate_term_id(label)

        # Add label metadata
        label_metadata = {
            "label": label,
            "concept_type": concept_type,
            "type": "ontology_label",
            **(metadata or {})
        }

        return self.ontology_labels.add(
            item_id=label_id,
            embedding=embedding,
            metadata=label_metadata,
            document=label
        )

    def add_external_ontology_label(self,
                                    label: str,
                                    embedding: List[float],
                                    source_ontology: str,
                                    metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add external ontology label for cross-ontology matching
        
        Args:
            label: The external ontology label
            embedding: Vector embedding for the label
            source_ontology: Source ontology name
            metadata: Optional metadata
            
        Returns:
            True if successful, False otherwise
        """
        label_id = self._generate_term_id(f"{source_ontology}:{label}")

        # Add external label metadata
        label_metadata = {
            "label": label,
            "source_ontology": source_ontology,
            "type": "external_ontology_label",
            **(metadata or {})
        }

        return self.external_ontology_labels.add(
            item_id=label_id,
            embedding=embedding,
            metadata=label_metadata,
            document=label
        )

    def find_similar_terms(self,
                           query_embedding: List[float],
                           collection_type: str = "candidate",
                           top_k: int = 10,
                           similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar terms in specified collection
        
        Args:
            query_embedding: Query vector embedding
            collection_type: Type of collection ("candidate", "ontology", "external")
            top_k: Number of results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of similar terms with similarity scores
        """
        # Select appropriate collection
        if collection_type == "candidate":
            operations = self.candidate_terms
        elif collection_type == "ontology":
            operations = self.ontology_labels
        elif collection_type == "external":
            operations = self.external_ontology_labels
        else:
            logger.error(f"Unknown collection type: {collection_type}")
            return []

        # Search for similar terms
        results = operations.search(
            query_embedding=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_documents=True
        )

        # Filter by similarity threshold and format results
        similar_terms = []
        for result in results:
            # Convert distance to similarity (assuming cosine distance)
            similarity = 1 - result["distance"]

            if similarity >= similarity_threshold:
                similar_terms.append({
                    "term": result["metadata"].get("term") or result["metadata"].get("label"),
                    "similarity": similarity,
                    "metadata": result["metadata"],
                    "document": result["document"]
                })

        logger.debug(f"Found {len(similar_terms)} similar terms in {collection_type} collection")
        return similar_terms

    def cluster_candidate_terms(self,
                                similarity_threshold: float = 0.8,
                                min_cluster_size: int = 2) -> List[List[Dict[str, Any]]]:
        """Cluster candidate terms by similarity
        
        Args:
            similarity_threshold: Minimum similarity for clustering
            min_cluster_size: Minimum size for a valid cluster
            
        Returns:
            List of clusters, each containing similar terms
        """
        try:
            # Get all candidate terms
            all_terms = self.candidate_terms.get(
                item_ids=self._get_all_ids("candidate_terms"),
                include_metadata=True,
                include_documents=True
            )

            if len(all_terms) < min_cluster_size:
                logger.info("Not enough terms for clustering")
                return []

            # Simple clustering algorithm
            clusters = []
            used_terms = set()

            for i, term in enumerate(all_terms):
                if term["id"] in used_terms:
                    continue

                # Get embedding for this term (would need to be stored or re-computed)
                # For now, we'll use a placeholder approach
                cluster = [term]
                used_terms.add(term["id"])

                # Find similar terms for this cluster
                # This is a simplified version - in practice you'd need the actual embeddings
                for j, other_term in enumerate(all_terms):
                    if other_term["id"] in used_terms or i == j:
                        continue

                    # In a real implementation, you'd compute similarity between embeddings
                    # For now, we'll use a placeholder
                    if self._terms_are_similar(term, other_term, similarity_threshold):
                        cluster.append(other_term)
                        used_terms.add(other_term["id"])

                if len(cluster) >= min_cluster_size:
                    clusters.append(cluster)

            logger.info(f"Created {len(clusters)} clusters from {len(all_terms)} terms")
            return clusters

        except Exception as e:
            logger.error(f"Failed to cluster candidate terms: {e}")
            return []

    def find_ontology_matches(self,
                              candidate_embedding: List[float],
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matching ontology labels for a candidate term
        
        Args:
            candidate_embedding: Embedding of candidate term
            top_k: Number of matches to return
            
        Returns:
            List of matching ontology labels with similarity scores
        """
        # Search in both ontology and external ontology collections
        ontology_matches = self.find_similar_terms(
            query_embedding=candidate_embedding,
            collection_type="ontology",
            top_k=top_k
        )

        external_matches = self.find_similar_terms(
            query_embedding=candidate_embedding,
            collection_type="external",
            top_k=top_k
        )

        # Combine and sort by similarity
        all_matches = ontology_matches + external_matches
        all_matches.sort(key=lambda x: x["similarity"], reverse=True)

        return all_matches[:top_k]

    def clear_collection(self, collection_type: str) -> bool:
        """Clear specified collection
        
        Args:
            collection_type: Type of collection to clear
            
        Returns:
            True if successful, False otherwise
        """
        if collection_type == "candidate":
            return self.candidate_terms.clear()
        elif collection_type == "ontology":
            return self.ontology_labels.clear()
        elif collection_type == "external":
            return self.external_ontology_labels.clear()
        else:
            logger.error(f"Unknown collection type: {collection_type}")
            return False

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics for all collections
        
        Returns:
            Dictionary with collection names and counts
        """
        return {
            "candidate_terms": self.candidate_terms.count(),
            "ontology_labels": self.ontology_labels.count(),
            "external_ontology_labels": self.external_ontology_labels.count()
        }

    def _generate_term_id(self, term: str) -> str:
        """Generate consistent ID for a term
        
        Args:
            term: Term to generate ID for
            
        Returns:
            Generated ID string
        """
        return ChromaClient.generate_id(term, prefix="term_")

    def _get_all_ids(self, collection_name: str) -> List[str]:
        """Get all IDs from a collection
        
        Args:
            collection_name: Name of collection
            
        Returns:
            List of all IDs in collection
        """
        try:
            collection = self.chroma_client.get_or_create_collection(collection_name)
            results = collection.get()
            return results.get("ids", [])
        except Exception as e:
            logger.error(f"Failed to get IDs from {collection_name}: {e}")
            return []

    def _terms_are_similar(self, term1: Dict, term2: Dict, threshold: float) -> bool:
        """Check if two terms are similar (placeholder implementation)
        
        Args:
            term1: First term dictionary
            term2: Second term dictionary
            threshold: Similarity threshold
            
        Returns:
            True if terms are similar, False otherwise
        """
        # This is a placeholder - in practice you'd compute embedding similarity
        # For now, we'll use simple string similarity
        text1 = term1.get("document", "").lower()
        text2 = term2.get("document", "").lower()

        # Simple Jaccard similarity
        if not text1 or not text2:
            return False

        words1 = set(text1.split())
        words2 = set(text2.split())

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
