# Episodic to Semantic Evolution

The **EpisodicToSemanticEvolver** promotes stable facts and events from episodic memory to semantic memory based on confidence and temporal stability.

## Overview

This evolver implements the natural progression of episodic memories (personal experiences) into semantic knowledge (factual information) when events demonstrate consistent patterns and high confidence over time.

## When Evolution Triggers

### Confidence Threshold
- **Trigger**: Episodic events reach confidence threshold
- **Default Confidence**: 0.9 (90% confidence)
- **Minimum Age**: 3 days (configurable)
- **Frequency**: Periodic evaluation during evolution cycles

### Configuration
```json
{
  "evolution": {
    "episodic_to_semantic_confidence": 0.9,
    "episodic_to_semantic_days": 3
  }
}
```

## Evolution Process

1. **Stability Analysis**: Identify episodic events with high confidence
2. **Temporal Validation**: Ensure events meet minimum age requirement
3. **Pattern Recognition**: Detect consistent patterns across events
4. **Semantic Promotion**: Convert stable events to semantic facts
5. **Episodic Archival**: Archive original episodic memories

## Implementation Details

```python
class EpisodicToSemanticEvolver(Evolver):
    def evolve(self, memory, logger=None):
        confidence = self.config.get("episodic_to_semantic_confidence", 0.9)
        min_days = self.config.get("episodic_to_semantic_days", 3)
        
        stable_events = memory.episodic.get_stable_events(
            confidence=confidence, 
            min_days=min_days
        )
        
        for event in stable_events:
            memory.semantic.add(event)
            memory.episodic.archive(event)
```

## Stability Criteria

### High Confidence Events
- **Repeated Patterns**: Events occurring multiple times
- **Cross-Validation**: Information confirmed from multiple sources
- **Temporal Consistency**: Stable information over time
- **User Reinforcement**: Frequently accessed memories

### Quality Factors
- **Factual Content**: Clear, objective information
- **Relevance Score**: High importance rating
- **Context Independence**: Information useful across contexts
- **Accuracy Validation**: Confirmed through usage patterns

## Evolution Examples

### Personal Facts
```
Episodic: "I learned Python syntax on March 15th"
→ Semantic: "Python uses indentation for code blocks"
```

### Learned Knowledge
```
Episodic: "Meeting discussion about API design patterns"
→ Semantic: "REST APIs should use HTTP status codes correctly"
```

### Professional Skills
```
Episodic: "Successfully debugged memory leak in production"
→ Semantic: "Memory profiling tools help identify leaks"
```

## Advanced Features

### Confidence Boosting
- **Retrieval Frequency**: Often-accessed memories gain confidence
- **Cross-Reference**: Memories linked to other stable facts
- **User Feedback**: Explicit or implicit validation
- **Time Stability**: Consistent information over extended periods

### Semantic Enrichment
- **Category Assignment**: Automatic topic classification
- **Relationship Discovery**: Links to existing semantic knowledge
- **Abstraction Levels**: General principles from specific experiences
- **Knowledge Integration**: Merger with existing semantic facts

## Benefits

- **Knowledge Crystallization**: Converts experiences into reusable facts
- **Memory Efficiency**: Reduces redundancy between memory types
- **Learning Acceleration**: Builds semantic knowledge base
- **Cognitive Modeling**: Mirrors human learning processes

## Best Practices

1. **Confidence Calibration**: Tune thresholds based on domain
2. **Quality Assurance**: Validate promoted semantic facts
3. **Relationship Preservation**: Maintain episodic-semantic links
4. **Gradual Promotion**: Avoid over-eager conversion

## Configuration Examples

### Conservative Settings
```json
{
  "evolution": {
    "episodic_to_semantic_confidence": 0.95,
    "episodic_to_semantic_days": 7,
    "require_multiple_confirmations": true
  }
}
```

### Aggressive Learning
```json
{
  "evolution": {
    "episodic_to_semantic_confidence": 0.8,
    "episodic_to_semantic_days": 1,
    "enable_pattern_acceleration": true
  }
}
```

## Related Evolvers

- [Episodic Decay](./episodic-decay) - Episodic memory archival
- [Episodic to Zettel](./episodic-to-zettel) - Episodic to knowledge notes
- [Semantic Decay](./semantic-decay) - Semantic memory pruning
