"""
Hybrid BM25 + Embedding retrieval for entity matching.

rank fusion approach, this module combines:
- BM25: Fast lexical matching (good for exact/partial matches)
- Embedding: Semantic similarity (good for synonyms/paraphrases)

The combination provides better recall than either method alone.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports
_bm25_available = None
_sentence_transformers_available = None


def _is_bm25_available() -> bool:
    """Check if rank_bm25 is available."""
    global _bm25_available
    if _bm25_available is None:
        try:
            from rank_bm25 import BM25Okapi
            _bm25_available = True
        except ImportError:
            _bm25_available = False
            logger.debug("rank_bm25 not installed, BM25 retrieval disabled")
    return _bm25_available


def _is_sentence_transformers_available() -> bool:
    """Check if sentence-transformers is available."""
    global _sentence_transformers_available
    if _sentence_transformers_available is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformers_available = True
        except ImportError:
            _sentence_transformers_available = False
            logger.debug("sentence-transformers not installed, embedding retrieval disabled")
    return _sentence_transformers_available


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and embedding similarity.
    
    approach:
    - BM25 for lexical matching
    - Embedding for semantic similarity
    - Rank fusion to combine scores
    
    Usage:
        retriever = HybridRetriever(items=["Apple Inc.", "Microsoft", "Google"])
        results = retriever.retrieve("tech company apple", top_k=5)
        # Returns: [("Apple Inc.", 0.95), ("Microsoft", 0.3), ...]
    """
    
    def __init__(
        self,
        items: List[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        bm25_weight: float = 0.5,
        embedding_weight: float = 0.5
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            items: List of items to index
            embedding_model: SentenceTransformer model name
            bm25_weight: Weight for BM25 scores (0-1)
            embedding_weight: Weight for embedding scores (0-1)
        """
        self.items = items
        self.embedding_model_name = embedding_model
        self.bm25_weight = bm25_weight
        self.embedding_weight = embedding_weight
        
        # Normalize weights
        total = bm25_weight + embedding_weight
        self.bm25_weight = bm25_weight / total
        self.embedding_weight = embedding_weight / total
        
        # Initialize components
        self._bm25 = None
        self._embeddings = None
        self._embedding_model = None
        self._tokenized = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize BM25 and embeddings."""
        if not self.items:
            return
        
        # Initialize BM25
        if _is_bm25_available():
            from rank_bm25 import BM25Okapi
            self._tokenized = [text.lower().split() for text in self.items]
            self._bm25 = BM25Okapi(self._tokenized)
            logger.debug(f"Initialized BM25 with {len(self.items)} items")
        
        # Initialize embeddings
        if _is_sentence_transformers_available():
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self._embeddings = self._embedding_model.encode(
                self.items, 
                show_progress_bar=False
            )
            logger.debug(f"Generated embeddings for {len(self.items)} items")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k items using hybrid ranking.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of (item, score) tuples, sorted by score descending
        """
        if not self.items:
            return []
        
        # Get BM25 scores
        bm25_scores = self._get_bm25_scores(query)
        
        # Get embedding scores
        embedding_scores = self._get_embedding_scores(query)
        
        # Combine scores
        if bm25_scores is not None and embedding_scores is not None:
            combined_scores = (
                self.bm25_weight * bm25_scores + 
                self.embedding_weight * embedding_scores
            )
        elif bm25_scores is not None:
            combined_scores = bm25_scores
        elif embedding_scores is not None:
            combined_scores = embedding_scores
        else:
            # No retrieval available, return items in order
            return [(item, 1.0) for item in self.items[:top_k]]
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        # Return results
        results = [
            (self.items[i], float(combined_scores[i]))
            for i in top_indices
        ]
        
        return results
    
    def _get_bm25_scores(self, query: str) -> Optional[np.ndarray]:
        """Get BM25 scores for query."""
        if self._bm25 is None:
            return None
        
        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)
        
        # Normalize to 0-1
        max_score = max(scores) if max(scores) > 0 else 1
        return np.array(scores) / max_score
    
    def _get_embedding_scores(self, query: str) -> Optional[np.ndarray]:
        """Get embedding similarity scores for query."""
        if self._embedding_model is None or self._embeddings is None:
            return None
        
        from sklearn.metrics.pairwise import cosine_similarity
        
        query_embedding = self._embedding_model.encode([query], show_progress_bar=False)
        scores = cosine_similarity(query_embedding, self._embeddings).flatten()
        
        # Scores are already 0-1 for cosine similarity
        return scores
    
    def get_relevant_items(
        self,
        query: str,
        top_k: int = 50,
        threshold: float = 0.0
    ) -> List[str]:
        """
        Get relevant items above threshold.
        
        Args:
            query: Query string
            top_k: Maximum items to return
            threshold: Minimum score threshold
            
        Returns:
            List of relevant items
        """
        results = self.retrieve(query, top_k)
        return [item for item, score in results if score >= threshold]


class EntityMatcher:
    """
    Entity matcher using hybrid retrieval.
    
    Useful for finding similar entities in a knowledge graph,
    e.g., for entity resolution or deduplication.
    """
    
    def __init__(
        self,
        entities: List[Dict[str, Any]],
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize entity matcher.
        
        Args:
            entities: List of entity dicts with 'name' key
            embedding_model: SentenceTransformer model name
        """
        self.entities = entities
        self.entity_names = [
            e.get('name') or e.get('content', '') 
            for e in entities
        ]
        
        self._retriever = HybridRetriever(
            items=self.entity_names,
            embedding_model=embedding_model
        )
    
    def find_similar(
        self,
        entity_name: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Find similar entities.
        
        Args:
            entity_name: Entity name to match
            top_k: Number of results
            exclude_self: Whether to exclude exact match
            
        Returns:
            List of (entity, score) tuples
        """
        results = self._retriever.retrieve(entity_name, top_k + (1 if exclude_self else 0))
        
        matched = []
        for name, score in results:
            if exclude_self and name.lower() == entity_name.lower():
                continue
            
            # Find entity by name
            idx = self.entity_names.index(name) if name in self.entity_names else -1
            if idx >= 0:
                matched.append((self.entities[idx], score))
        
        return matched[:top_k]
    
    def find_matches(
        self,
        entity_name: str,
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Find entities matching above threshold.
        
        Args:
            entity_name: Entity name to match
            threshold: Minimum similarity score
            
        Returns:
            List of matching entities
        """
        results = self.find_similar(entity_name, top_k=50)
        return [entity for entity, score in results if score >= threshold]


def hybrid_search(
    query: str,
    items: List[str],
    top_k: int = 10,
    bm25_weight: float = 0.5,
    embedding_weight: float = 0.5,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> List[Tuple[str, float]]:
    """
    Convenience function for hybrid search.
    
    Args:
        query: Search query
        items: Items to search
        top_k: Number of results
        bm25_weight: Weight for BM25
        embedding_weight: Weight for embeddings
        embedding_model: Embedding model name
        
    Returns:
        List of (item, score) tuples
    """
    retriever = HybridRetriever(
        items=items,
        embedding_model=embedding_model,
        bm25_weight=bm25_weight,
        embedding_weight=embedding_weight
    )
    return retriever.retrieve(query, top_k)
