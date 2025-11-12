# Working to Episodic Evolution

The **WorkingToEpisodicEvolver** manages the consolidation of working memory into episodic memory when the working buffer reaches capacity.

## Overview

This evolver implements the transition from immediate working memory to longer-term episodic storage, mimicking human cognitive patterns where working memory items are either forgotten or consolidated into episodic memories.

## When Evolution Triggers

### Buffer Overflow
- **Trigger**: Working memory buffer exceeds threshold
- **Default Threshold**: 40 items (configurable)
- **Frequency**: Immediate when threshold is reached
- **Automatic**: Yes

### Configuration
```json
{
  "evolution": {
    "working_to_episodic_threshold": 40
  }
}
```

## Evolution Process

1. **Buffer Check**: Monitor working memory buffer size
2. **Threshold Evaluation**: Compare current size to configured threshold
3. **Summarization**: Create consolidated summary of working items
4. **Episodic Storage**: Store summary in episodic memory
5. **Buffer Clear**: Clear working memory buffer

## Implementation Details

```python
class WorkingToEpisodicEvolver(Evolver):
    def evolve(self, memory, logger=None):
        threshold = self.config.get("working_to_episodic_threshold", 40)
        working_items = memory.working.get_buffer()
        
        if len(working_items) >= threshold:
            summary = memory.working.summarize_buffer()
            memory.episodic.add(summary)
            memory.working.clear_buffer()
```

## Enhanced Version

The **EnhancedWorkingToEpisodicEvolver** provides advanced features:

### Key Improvements
- **Adaptive Thresholds**: Dynamic capacity based on cognitive load
- **Semantic Clustering**: Groups related items before summarization
- **Temporal Weighting**: Prioritizes recent items
- **Context-Aware Grouping**: Uses entity overlap and semantic similarity

### Enhanced Configuration
```json
{
  "evolution": {
    "working_to_episodic_threshold": 40,
    "adaptive_threshold": true,
    "semantic_clustering": true,
    "temporal_decay_factor": 0.1,
    "cluster_similarity_threshold": 0.7
  }
}
```

### Enhanced Process
1. **Cognitive Load Assessment**: Evaluate current memory pressure
2. **Adaptive Threshold**: Adjust capacity based on load
3. **Semantic Clustering**: Group related items using similarity
4. **Temporal Weighting**: Apply decay to older items
5. **Enhanced Summarization**: Create context-aware summaries
6. **Selective Evolution**: Preserve high-priority working items

## Benefits

- **Memory Efficiency**: Prevents working memory overflow
- **Knowledge Preservation**: Maintains important information
- **Cognitive Modeling**: Mimics human memory consolidation
- **Adaptive Behavior**: Responds to memory pressure

## Best Practices

1. **Threshold Tuning**: Adjust based on usage patterns
2. **Summary Quality**: Ensure meaningful consolidation
3. **Context Preservation**: Maintain important relationships
4. **Performance Monitoring**: Track evolution effectiveness

## Related Evolvers

- [Exponential Decay](./exponential-decay) - Scientific forgetting patterns
- [Working to Procedural](./working-to-procedural) - Skill pattern evolution
- [Episodic Decay](./episodic-decay) - Episodic memory pruning
