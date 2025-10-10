"""
ChromaDB Integration for Ontology Semantic Clustering

Handles vector embeddings for concept/relation similarity and clustering.
"""

import hashlib
import json
import logging
from chromadb import Settings
from typing import Dict, List, Optional, Tuple, Any

from smartmemory.configuration import MemoryConfig
from smartmemory.ontology.ir_models import Concept, Relation, Status, Origin, Meta

logger = logging.getLogger(__name__)


class ChromaOntologyService:
    """ChromaDB service for ontology semantic operations"""

    def __init__(self, config: MemoryConfig = None):
        if config is None:
            config = MemoryConfig()

        self.config = config

        # Use config for ChromaDB connection
        host = config.vector.get("host", "localhost")
        port = config.vector.get("port", 8000)

        try:
            import chromadb

            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(anonymized_telemetry=False)
            )

            # Collections for different types
            self.candidate_terms = self._get_or_create_collection("candidate_terms")
            self.ontology_labels = self._get_or_create_collection("ontology_labels")
            self.external_ontology_labels = self._get_or_create_collection("external_ontology_labels")

            logger.info("ChromaDB initialized for ontology clustering")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None

    def _get_or_create_collection(self, name: str):
        """Get or create ChromaDB collection"""
        if not self.client:
            return None

        try:
            return self.client.get_or_create_collection(name=name)
        except Exception as e:
            logger.error(f"Failed to create collection {name}: {e}")
            return None

    def find_similar_concepts(
            self,
            concept: Concept,
            threshold: float = 0.80,
            limit: int = 10
    ) -> List[Tuple[Concept, float]]:
        """Find similar concepts using vector similarity"""
        if not self.client or not self.candidate_terms:
            return self._fallback_similarity(concept, threshold, limit)

        try:
            # Query for similar concepts
            results = self.candidate_terms.query(
                query_texts=[concept.label],
                n_results=limit,
                where={"status": {"$ne": "merged"}}  # Exclude already merged
            )

            similar_concepts = []
            if results and results['documents']:
                for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    similarity = 1.0 - distance  # Convert distance to similarity
                    if similarity >= threshold:
                        # Reconstruct concept from metadata
                        metadata = results['metadatas'][0][i]
                        similar_concept = self._metadata_to_concept(metadata)
                        similar_concepts.append((similar_concept, similarity))

            return similar_concepts

        except Exception as e:
            logger.error(f"ChromaDB similarity search failed: {e}")
            return self._fallback_similarity(concept, threshold, limit)

    def find_similar_relations(
            self,
            relation: Relation,
            threshold: float = 0.80,
            limit: int = 10
    ) -> List[Tuple[Relation, float]]:
        """Find similar relations using vector similarity"""
        if not self.client or not self.ontology_labels:
            return []

        try:
            # Query for similar relations
            results = self.ontology_labels.query(
                query_texts=[relation.label],
                n_results=limit,
                where={"type": "relation", "status": {"$ne": "merged"}}
            )

            similar_relations = []
            if results and results['documents']:
                for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    similarity = 1.0 - distance
                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        similar_relation = self._metadata_to_relation(metadata)
                        similar_relations.append((similar_relation, similarity))

            return similar_relations

        except Exception as e:
            logger.error(f"ChromaDB relation similarity failed: {e}")
            return []

    def cluster_concepts(
            self,
            concepts: List[Concept],
            similarity_threshold: float = 0.80
    ) -> List[List[Concept]]:
        """Cluster concepts by semantic similarity"""
        if not self.client:
            return self._fallback_clustering(concepts, similarity_threshold)

        try:
            # Add concepts to collection for clustering
            self._add_concepts_to_collection(concepts)

            # Perform clustering using similarity search
            clusters = []
            processed = set()

            for concept in concepts:
                if concept.id in processed:
                    continue

                # Find similar concepts
                similar = self.find_similar_concepts(concept, similarity_threshold)

                # Create cluster
                cluster = [concept]
                processed.add(concept.id)

                for similar_concept, score in similar:
                    if similar_concept.id not in processed:
                        cluster.append(similar_concept)
                        processed.add(similar_concept.id)

                if len(cluster) > 1:
                    clusters.append(cluster)
                else:
                    # Single concept clusters
                    clusters.append([concept])

            return clusters

        except Exception as e:
            logger.error(f"ChromaDB clustering failed: {e}")
            return self._fallback_clustering(concepts, similarity_threshold)

    def _add_concepts_to_collection(self, concepts: List[Concept]):
        """Add concepts to ChromaDB collection"""
        if not self.candidate_terms:
            return

        try:
            documents = []
            metadatas = []
            ids = []

            for concept in concepts:
                # Use concept text for embedding
                text = f"{concept.label} {' '.join(concept.synonyms)}"
                documents.append(text)

                # Store concept metadata
                metadata = {
                    "id": concept.id,
                    "label": concept.label,
                    "synonyms": json.dumps(concept.synonyms),
                    "status": concept.status.value,
                    "confidence": concept.confidence,
                    "type": "concept"
                }
                metadatas.append(metadata)

                # Use concept ID as ChromaDB ID
                ids.append(self._safe_id(concept.id))

            # Add to collection (upsert to handle duplicates)
            self.candidate_terms.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

        except Exception as e:
            logger.error(f"Failed to add concepts to ChromaDB: {e}")

    def add_external_ontology(self, ontology_name: str, concepts: List[Dict[str, Any]]):
        """Add external ontology concepts for alignment"""
        if not self.external_ontology_labels:
            return

        try:
            documents = []
            metadatas = []
            ids = []

            for concept in concepts:
                text = f"{concept.get('label', '')} {' '.join(concept.get('synonyms', []))}"
                documents.append(text)

                metadata = {
                    "ontology": ontology_name,
                    "curie": concept.get("curie", ""),
                    "label": concept.get("label", ""),
                    "description": concept.get("description", ""),
                    "type": "external_concept"
                }
                metadatas.append(metadata)

                # Use ontology:curie as ID
                ids.append(self._safe_id(f"{ontology_name}:{concept.get('curie', '')}"))

            self.external_ontology_labels.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(concepts)} concepts from {ontology_name}")

        except Exception as e:
            logger.error(f"Failed to add external ontology {ontology_name}: {e}")

    def find_alignment_candidates(
            self,
            concept: Concept,
            ontology_name: Optional[str] = None,
            limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find external ontology alignment candidates"""
        if not self.external_ontology_labels:
            return []

        try:
            where_clause = {"type": "external_concept"}
            if ontology_name:
                where_clause["ontology"] = ontology_name

            results = self.external_ontology_labels.query(
                query_texts=[concept.label],
                n_results=limit,
                where=where_clause
            )

            candidates = []
            if results and results['documents']:
                for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
                    similarity = 1.0 - distance
                    metadata = results['metadatas'][0][i]

                    candidate = {
                        "curie": metadata.get("curie", ""),
                        "source": metadata.get("ontology", ""),
                        "label": metadata.get("label", ""),
                        "description": metadata.get("description", ""),
                        "confidence": similarity
                    }
                    candidates.append(candidate)

            return candidates

        except Exception as e:
            logger.error(f"Alignment search failed: {e}")
            return []

    def _metadata_to_concept(self, metadata: Dict[str, Any]) -> Concept:
        """Convert ChromaDB metadata back to Concept"""
        from datetime import datetime

        synonyms = []
        try:
            synonyms = json.loads(metadata.get("synonyms", "[]"))
        except:
            pass

        return Concept(
            id=metadata.get("id", ""),
            label=metadata.get("label", ""),
            synonyms=synonyms,
            status=Status(metadata.get("status", "proposed")),
            origin=Origin.AI,
            confidence=metadata.get("confidence", 0.0),
            meta=Meta(created_by="chroma", created_at=datetime.now())
        )

    def _metadata_to_relation(self, metadata: Dict[str, Any]) -> Relation:
        """Convert ChromaDB metadata back to Relation"""

        return Relation(
            id=metadata.get("id", ""),
            label=metadata.get("label", ""),
            status=Status(metadata.get("status", "proposed")),
            confidence=metadata.get("confidence", 0.0)
        )

    def _safe_id(self, id_str: str) -> str:
        """Create safe ChromaDB ID"""
        # ChromaDB IDs must be strings and unique
        return hashlib.md5(id_str.encode()).hexdigest()

    def _fallback_similarity(
            self,
            concept: Concept,
            threshold: float,
            limit: int
    ) -> List[Tuple[Concept, float]]:
        """Fallback similarity using simple string matching"""
        # Simple edit distance fallback

        # This would need access to other concepts - simplified for now
        return []

    def _fallback_clustering(
            self,
            concepts: List[Concept],
            threshold: float
    ) -> List[List[Concept]]:
        """Fallback clustering using string similarity"""
        import difflib

        clusters = []
        processed = set()

        for concept in concepts:
            if concept.id in processed:
                continue

            cluster = [concept]
            processed.add(concept.id)

            # Find similar concepts using string similarity
            for other in concepts:
                if other.id in processed:
                    continue

                similarity = difflib.SequenceMatcher(
                    None, concept.label.lower(), other.label.lower()
                ).ratio()

                if similarity >= threshold:
                    cluster.append(other)
                    processed.add(other.id)

            clusters.append(cluster)

        return clusters
