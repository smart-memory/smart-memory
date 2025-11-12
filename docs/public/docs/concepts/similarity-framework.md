# Similarity Framework

SmartMemory's similarity framework provides sophisticated methods for measuring relationships between memories, enabling intelligent retrieval, linking, and organization.

## Overview

The similarity framework combines multiple similarity metrics to create a comprehensive understanding of memory relationships, supporting both semantic and structural similarity analysis.

## Similarity Metrics

### Semantic Similarity

**Vector-Based Similarity:**
- **Cosine Similarity** - Angular distance between embeddings
- **Euclidean Distance** - Geometric distance in vector space
- **Dot Product** - Direct vector multiplication
- **Manhattan Distance** - Sum of absolute differences

**Implementation:**
```python
# Configure semantic similarity
similarity_config = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "similarity_metric": "cosine",
    "threshold": 0.7
}

# Calculate similarity
similarity = memory.calculate_similarity(memory1, memory2, metric="semantic")
```

### Temporal Similarity

**Time-Based Relationships:**
- **Temporal Proximity** - Closeness in time
- **Sequence Similarity** - Order of events
- **Duration Overlap** - Time period intersection
- **Frequency Patterns** - Recurring temporal patterns

**Features:**
- Adaptive time windows
- Context-aware weighting
- Seasonal pattern recognition
- Event sequence analysis

### Structural Similarity

**Graph-Based Metrics:**
- **Neighborhood Similarity** - Shared connections
- **Path Similarity** - Common relationship paths
- **Centrality Similarity** - Similar importance in network
- **Clustering Coefficient** - Local connectivity patterns

### Content Similarity

**Text-Based Analysis:**
- **N-gram Overlap** - Shared word sequences
- **TF-IDF Similarity** - Term frequency analysis
- **Jaccard Similarity** - Set intersection over union
- **Edit Distance** - Character-level differences

## Multi-Dimensional Similarity

### Weighted Combination

```python
similarity_weights = {
    "semantic": 0.4,
    "temporal": 0.3,
    "structural": 0.2,
    "content": 0.1
}

combined_similarity = memory.calculate_weighted_similarity(
    memory1, memory2, 
    weights=similarity_weights
)
```

### Adaptive Weighting

**Context-Dependent Weights:**
- **Query Type** - Adjust weights based on search intent
- **Memory Type** - Different weights for different memory types
- **User Preferences** - Learn from user interactions
- **Domain Specific** - Optimize for content domain

### Dynamic Thresholds

**Adaptive Thresholds:**
- **Statistical Analysis** - Data-driven threshold selection
- **Performance Optimization** - Optimize for retrieval quality
- **User Feedback** - Adjust based on relevance ratings
- **Context Sensitivity** - Different thresholds for different contexts

## Similarity Applications

### Memory Linking

**Automatic Relationship Discovery:**
```python
# Find similar memories
similar_memories = memory.find_similar(
    target_memory,
    similarity_threshold=0.8,
    max_results=10
)

# Create explicit links
for similar_memory in similar_memories:
    memory.create_link(
        target_memory.id,
        similar_memory.id,
        relationship_type="similar",
        strength=similar_memory.similarity_score
    )
```

### Clustering and Organization

**Memory Clustering:**
- **Hierarchical Clustering** - Tree-like organization
- **K-Means Clustering** - Fixed number of clusters
- **DBSCAN** - Density-based clustering
- **Community Detection** - Graph-based communities

### Search and Retrieval

**Similarity-Based Search:**
```python
# Semantic search
results = memory.search(
    query="machine learning concepts",
    search_type="semantic",
    similarity_threshold=0.6
)

# Multi-modal search
results = memory.search(
    query="recent project meetings",
    search_types=["semantic", "temporal"],
    weights={"semantic": 0.7, "temporal": 0.3}
)
```

## Advanced Features

### Similarity Learning

**Machine Learning Enhancement:**
- **Metric Learning** - Learn optimal similarity functions
- **Deep Similarity Networks** - Neural network-based similarity
- **Transfer Learning** - Adapt pre-trained models
- **Active Learning** - Improve with user feedback

### Cross-Modal Similarity

**Multi-Modal Comparison:**
- **Text-Image Similarity** - Compare textual and visual content
- **Audio-Text Similarity** - Speech and text comparison
- **Temporal-Spatial Similarity** - Time and location relationships
- **Structured-Unstructured** - Database and text comparison

### Similarity Caching

**Performance Optimization:**
```python
# Enable similarity caching
memory.enable_similarity_cache(
    cache_size="1GB",
    ttl="24h",
    update_strategy="lazy"
)

# Precompute similarities
memory.precompute_similarities(
    memory_set=recent_memories,
    similarity_types=["semantic", "temporal"]
)
```

## Configuration and Tuning

### Similarity Profiles

```python
# Create custom similarity profile
profile = SimilarityProfile(
    name="research_assistant",
    weights={
        "semantic": 0.5,
        "temporal": 0.2,
        "structural": 0.2,
        "content": 0.1
    },
    thresholds={
        "high_similarity": 0.8,
        "medium_similarity": 0.6,
        "low_similarity": 0.4
    }
)

memory.set_similarity_profile(profile)
```

### Performance Tuning

**Optimization Strategies:**
- **Approximate Similarity** - Trade accuracy for speed
- **Hierarchical Search** - Coarse-to-fine similarity
- **Parallel Processing** - Multi-threaded computation
- **GPU Acceleration** - Hardware-accelerated similarity

### Evaluation Metrics

**Similarity Quality Assessment:**
- **Precision@K** - Relevant results in top K
- **Recall** - Coverage of relevant memories
- **F1-Score** - Harmonic mean of precision and recall
- **Mean Average Precision** - Average precision across queries

The similarity framework provides the foundation for intelligent memory organization and retrieval, enabling SmartMemory to understand complex relationships and provide relevant, contextual responses to user queries.
