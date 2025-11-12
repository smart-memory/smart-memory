# Episodic Decay Evolution

The **EpisodicDecayEvolver** manages the archival and removal of stale episodic events based on age and relevance to maintain optimal memory performance.

## Overview

This evolver implements temporal decay for episodic memories, automatically archiving or removing events that have become irrelevant over time, mimicking natural forgetting processes while preserving important memories.

## When Evolution Triggers

### Time-Based Decay
- **Trigger**: Events exceed configured half-life
- **Default Half-Life**: 30 days (configurable)
- **Frequency**: Periodic evaluation during evolution cycles
- **Automatic**: Yes, part of maintenance routines

### Configuration
```json
{
  "evolution": {
    "episodic_decay_half_life": 30,
    "decay_strategy": "archive",
    "preserve_high_importance": true,
    "minimum_access_for_preservation": 3
  }
}
```

## Evolution Process

1. **Age Analysis**: Evaluate episodic events against time thresholds
2. **Relevance Assessment**: Check event importance and access patterns
3. **Decay Calculation**: Apply exponential decay based on half-life
4. **Preservation Check**: Identify events worth preserving
5. **Archival Execution**: Move stale events to archive or cold storage

## Implementation Details

```python
class EpisodicDecayEvolver(Evolver):
    def evolve(self, memory, logger=None):
        half_life = self.config.get("episodic_decay_half_life", 30)  # days
        stale_events = memory.episodic.get_stale_events(half_life=half_life)
        
        for event in stale_events:
            memory.episodic.archive(event)
            if logger:
                logger.info(f"Archived stale episodic event: {event}")
```

## Decay Strategies

### Archive Strategy
- **Soft Removal**: Move to archive storage
- **Metadata Preservation**: Maintain event metadata
- **Retrieval Possibility**: Events can be restored if needed
- **Space Optimization**: Reduce active memory footprint

### Deletion Strategy
- **Hard Removal**: Permanently delete events
- **Space Recovery**: Free storage completely
- **No Recovery**: Cannot be undone
- **Performance Focus**: Maximum memory optimization

### Selective Preservation
- **High Importance**: Preserve critical events regardless of age
- **Frequent Access**: Keep often-referenced events
- **Linked Events**: Preserve events with many connections
- **User Bookmarks**: Respect explicit user preferences

## Decay Factors

### Time-Based Decay
- **Exponential Curve**: Natural forgetting pattern
- **Half-Life Model**: Configurable decay rate
- **Age Weighting**: Older events decay faster
- **Recent Protection**: Recent events are preserved

### Usage-Based Preservation
- **Access Frequency**: Recently accessed events resist decay
- **Reference Count**: Events linked by other memories
- **Search Appearances**: Events appearing in search results
- **User Interactions**: Events with user engagement

### Content-Based Factors
- **Importance Score**: High-value events preserved longer
- **Uniqueness**: Rare or unique events get protection
- **Learning Value**: Educational content preserved
- **Emotional Significance**: Emotionally tagged events

## Examples

### Routine Events (Quick Decay)
```
Event: "Daily standup meeting notes from 2 months ago"
→ Archived after 30 days (routine, low uniqueness)
```

### Important Decisions (Preserved)
```
Event: "Architecture decision: chose microservices over monolith"
→ Preserved indefinitely (high importance, frequently referenced)
```

### Learning Moments (Selective)
```
Event: "Learned about React hooks in tutorial"
→ Converted to semantic knowledge, episodic archived
```

## Advanced Decay Models

### Exponential Decay
```python
decay_strength = exp(-age_days / half_life)
preservation_probability = decay_strength * importance_factor
```

### Usage-Boosted Decay
```python
usage_boost = log(1 + access_count) * recency_factor
adjusted_decay = base_decay - usage_boost
```

### Context-Sensitive Decay
```python
context_relevance = similarity_to_current_context()
decay_rate = base_decay_rate * (1 - context_relevance)
```

## Benefits

- **Memory Efficiency**: Prevents episodic memory bloat
- **Performance Optimization**: Faster retrieval of relevant events
- **Natural Forgetting**: Mimics human memory patterns
- **Focus Enhancement**: Emphasizes important memories

## Configuration Examples

### Conservative Decay
```json
{
  "evolution": {
    "episodic_decay_half_life": 90,
    "decay_strategy": "archive",
    "preserve_linked_events": true,
    "minimum_importance_for_preservation": 0.3
  }
}
```

### Aggressive Cleanup
```json
{
  "evolution": {
    "episodic_decay_half_life": 14,
    "decay_strategy": "delete",
    "preserve_only_bookmarked": true,
    "fast_decay_for_routine": true
  }
}
```

### Learning-Focused
```json
{
  "evolution": {
    "episodic_decay_half_life": 45,
    "preserve_learning_events": true,
    "convert_to_semantic": true,
    "learning_event_threshold": 0.7
  }
}
```

## Best Practices

1. **Gradual Implementation**: Start with conservative settings
2. **User Feedback**: Monitor user satisfaction with decay
3. **Recovery Mechanisms**: Provide event restoration capabilities
4. **Importance Calibration**: Tune importance scoring algorithms
5. **Performance Monitoring**: Track memory usage and retrieval speed

## Integration Features

### Cross-Memory Coordination
- **Semantic Promotion**: Convert valuable events before decay
- **Zettel Creation**: Transform insights into permanent notes
- **Procedural Learning**: Extract skill patterns before archival
- **Working Memory**: Respect working memory references

### User Control
- **Manual Override**: Users can prevent decay of specific events
- **Importance Adjustment**: Users can modify event importance
- **Decay Settings**: Per-user decay preferences
- **Recovery Tools**: Restore accidentally archived events

## Related Evolvers

- [Episodic to Semantic](./episodic-to-semantic) - Event promotion before decay
- [Episodic to Zettel](./episodic-to-zettel) - Note creation from events
- [Semantic Decay](./semantic-decay) - Semantic memory maintenance
