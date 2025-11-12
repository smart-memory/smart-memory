# Zettel Pruning Evolution

The **ZettelPruneEvolver** maintains Zettelkasten health by pruning and merging low-quality or duplicate Zettel notes to ensure graph quality and coherence.

## Overview

This evolver implements quality control for the Zettelkasten knowledge graph, identifying and removing or merging notes that reduce overall graph quality, maintaining an optimal knowledge network.

## When Evolution Triggers

### Quality-Based Triggers
- **Low Quality Detection**: Notes falling below quality thresholds
- **Duplicate Detection**: Nearly identical notes identified
- **Orphan Detection**: Notes with no connections
- **Frequency**: Part of periodic maintenance cycles

### Configuration
```json
{
  "evolution": {
    "zettel_pruning": {
      "enable_duplicate_detection": true,
      "enable_quality_pruning": true,
      "enable_orphan_removal": true,
      "similarity_threshold": 0.95,
      "minimum_connection_count": 1,
      "quality_score_threshold": 0.3
    }
  }
}
```

## Evolution Process

1. **Quality Analysis**: Evaluate all Zettel notes for quality metrics
2. **Duplicate Detection**: Identify highly similar or redundant notes
3. **Orphan Identification**: Find isolated notes without connections
4. **Merge Strategy**: Combine duplicates preserving best content
5. **Pruning Execution**: Remove or archive low-quality notes

## Implementation Details

```python
class ZettelPruneEvolver(Evolver):
    def evolve(self, memory, logger=None):
        zettels = memory.zettel.get_low_quality_or_duplicates()
        
        for z in zettels:
            memory.zettel.prune_or_merge(z)
            if logger:
                logger.info(f"Pruned/merged zettel: {z}")
```

## Quality Assessment Criteria

### Content Quality
- **Atomicity**: Single coherent idea per note
- **Clarity**: Clear and understandable content
- **Completeness**: Sufficient detail for usefulness
- **Accuracy**: Factual correctness and reliability

### Structural Quality
- **Link Density**: Number and quality of connections
- **Tag Relevance**: Appropriate and meaningful tags
- **Metadata Completeness**: Required metadata fields present
- **Graph Integration**: Proper integration with knowledge network

### Usage Patterns
- **Access Frequency**: How often note is referenced
- **Link Traversal**: Usage in knowledge navigation
- **Update Frequency**: How often content is modified
- **Search Relevance**: Appearance in search results

## Duplicate Detection

### Similarity Measures
- **Content Similarity**: Text-based comparison using embeddings
- **Structural Similarity**: Tag and metadata overlap
- **Entity Overlap**: Shared entities and concepts
- **Link Pattern**: Similar connection patterns

### Merge Strategies
1. **Content Union**: Combine all unique information
2. **Best Quality**: Keep highest quality version
3. **Most Connected**: Preserve note with most links
4. **Recent Priority**: Favor more recently updated content

### Example Merge Process
```
Note A: "JavaScript arrow functions provide concise syntax"
Note B: "Arrow functions in JS offer shorter syntax for functions"

Merged: "JavaScript arrow functions provide concise syntax for function definitions"
+ Combined tags: #javascript #functions #syntax #es6
+ Preserved all links from both notes
+ Updated metadata with merge history
```

## Orphan Management

### Orphan Identification
- **Zero Connections**: Notes with no incoming or outgoing links
- **Weak Connections**: Notes with only low-quality connections
- **Isolated Clusters**: Small disconnected groups

### Orphan Strategies
1. **Auto-Linking**: Attempt to find relevant connections
2. **Tag Enhancement**: Add tags to improve discoverability
3. **Content Review**: Flag for manual review
4. **Archival**: Move to archive if truly isolated

## Pruning Categories

### Immediate Removal
- **Empty Notes**: Notes with no meaningful content
- **Duplicate Exact**: Identical content and structure
- **Broken Notes**: Corrupted or invalid format
- **Spam Content**: Obviously inappropriate content

### Conditional Removal
- **Low Usage**: Rarely accessed notes below threshold
- **Poor Quality**: Notes scoring low on quality metrics
- **Outdated Information**: Facts superseded by newer knowledge
- **Redundant Detail**: Overly specific notes with broader alternatives

### Archive Candidates
- **Historical Notes**: Old but potentially valuable
- **Context-Specific**: Useful only in specific contexts
- **Personal Notes**: Private or highly individual content
- **Work-in-Progress**: Incomplete but developing notes

## Benefits

- **Graph Health**: Maintains clean, high-quality knowledge network
- **Navigation Efficiency**: Removes noise from knowledge exploration
- **Storage Optimization**: Reduces redundant content storage
- **Search Quality**: Improves relevance of search results

## Configuration Examples

### Conservative Pruning
```json
{
  "evolution": {
    "zettel_pruning": {
      "similarity_threshold": 0.98,
      "minimum_connection_count": 0,
      "quality_score_threshold": 0.1,
      "require_manual_review": true
    }
  }
}
```

### Aggressive Cleanup
```json
{
  "evolution": {
    "zettel_pruning": {
      "similarity_threshold": 0.85,
      "minimum_connection_count": 2,
      "quality_score_threshold": 0.5,
      "auto_merge_duplicates": true
    }
  }
}
```

## Best Practices

1. **Backup Strategy**: Always backup before major pruning operations
2. **Gradual Approach**: Start with conservative settings
3. **Review Process**: Include manual review for edge cases
4. **Quality Metrics**: Establish clear quality criteria
5. **User Feedback**: Incorporate user preferences and patterns

## Monitoring and Metrics

### Health Indicators
- **Graph Density**: Connection ratio after pruning
- **Quality Distribution**: Score distribution across notes
- **Usage Patterns**: Access patterns post-pruning
- **User Satisfaction**: Feedback on pruning results

### Performance Metrics
- **Pruning Rate**: Percentage of notes pruned per cycle
- **Merge Success**: Quality of merged note combinations
- **False Positives**: Incorrectly pruned valuable notes
- **Graph Coherence**: Overall knowledge graph integrity

## Related Evolvers

- [Episodic to Zettel](./episodic-to-zettel) - Zettel creation from events
- [Maximal Connectivity](./maximal-connectivity) - Enhanced linking
- [Strategic Pruning](./strategic-pruning) - Cross-memory pruning
