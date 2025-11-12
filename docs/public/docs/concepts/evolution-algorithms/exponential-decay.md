# Exponential Decay Evolution

The **ExponentialDecayEvolver** implements scientifically-based exponential forgetting curves instead of linear decay, based on Ebbinghaus forgetting curve research.

## Overview

This enhanced evolver models how human memory naturally decays exponentially over time, with memory strength declining rapidly at first and then more slowly. Unlike linear decay, this approach more accurately reflects cognitive science research and can be restored through retrieval practice.

## When Evolution Triggers

### Time-Based Exponential Decay
- **Trigger**: Periodic evaluation of all memories
- **Decay Rate**: 0.1 per day (configurable)
- **Minimum Strength**: 0.1 threshold before archival
- **Frequency**: Continuous background evaluation

### Configuration
```json
{
  "evolution": {
    "exponential_decay_rate": 0.1,
    "min_strength_threshold": 0.1,
    "enable_retrieval_strengthening": true,
    "decay_check_interval_hours": 24
  }
}
```

## Evolution Process

1. **Memory Analysis**: Get all memories eligible for decay
2. **Decay Calculation**: Apply exponential decay formula
3. **Retrieval Strengthening**: Account for access-based strengthening
4. **Threshold Check**: Identify memories below minimum strength
5. **Archival Decision**: Archive or strengthen weak memories

## Implementation Details

```python
class ExponentialDecayEvolver(Evolver):
    def evolve(self, memory, logger=None):
        decay_rate = self.config.get("exponential_decay_rate", 0.1)  # per day
        min_strength = self.config.get("min_strength_threshold", 0.1)
        
        all_memories = self._get_decayable_memories(memory)
        
        for memory_item in all_memories:
            current_strength = self._calculate_decay_strength(memory_item, decay_rate)
            
            if current_strength < min_strength:
                self._archive_weak_memory(memory, memory_item)
```

## Scientific Foundation

### Ebbinghaus Forgetting Curve
The evolver implements the mathematical model:
```
R = e^(-t/S)
```
Where:
- **R** = Retention strength (0-1)
- **t** = Time since learning/last access
- **S** = Memory strength factor
- **e** = Euler's constant (natural exponential)

### Retrieval Practice Effect
Memory strength can be restored through access:
```
new_strength = base_strength * retrieval_boost
retrieval_boost = 1 + log(1 + access_count) * recency_factor
```

## Decay Calculation

### Base Decay Formula

```python
import smartmemory.utils


def calculate_decay_strength(self, memory_item, decay_rate):
    age_days = (smartmemory.utils.now(timezone.utc) - memory_item.created_at).days
    base_strength = math.exp(-age_days * decay_rate)

    # Account for retrieval strengthening
    retrieval_strength = memory_item.metadata.get('retrieval_strength', 1.0)

    return min(1.0, base_strength * retrieval_strength)
```

### Factors Affecting Decay
- **Age**: Older memories decay more (exponential curve)
- **Access Frequency**: Recent access slows decay
- **Importance**: High-importance memories resist decay
- **Quality**: Well-formed memories decay slower
- **Connections**: Linked memories support each other

## Retrieval Strengthening

### Access-Based Boosting
- **Recent Access**: Recently retrieved memories get strength boost
- **Frequency Effect**: Frequently accessed memories build resistance
- **Spaced Retrieval**: Optimal spacing between accesses
- **Context Reinforcement**: Retrieval in different contexts strengthens memory

### Strengthening Mechanisms
1. **Immediate Boost**: Instant strength increase on access
2. **Decay Resistance**: Slower future decay rate
3. **Connection Strengthening**: Enhanced links to related memories
4. **Quality Improvement**: Better encoding through retrieval

## Examples

### Research Paper Memory
```
Initial: Strength = 1.0 (just read)
After 1 day: Strength = 0.90 (e^(-1*0.1))
After 7 days: Strength = 0.50 (e^(-7*0.1))
After access: Strength = 0.75 (0.50 * 1.5 retrieval boost)
After 30 days: Strength = 0.05 → Archived
```

### Frequently Used Code Pattern
```
Initial: Strength = 1.0
After 1 day: Strength = 0.90
Access at day 2: Boost to 1.0, slower future decay
After 7 days: Strength = 0.70 (slowed by retrieval)
Regular access: Maintains high strength
```

## Benefits

- **Scientific Accuracy**: Based on cognitive science research
- **Natural Patterns**: Mimics human memory behavior
- **Retrieval Benefits**: Rewards memory access
- **Adaptive Decay**: Different rates for different memory types

## Configuration Examples

### Aggressive Forgetting
```json
{
  "evolution": {
    "exponential_decay_rate": 0.2,
    "min_strength_threshold": 0.2,
    "fast_decay_for_routine": true
  }
}
```

### Conservative Preservation
```json
{
  "evolution": {
    "exponential_decay_rate": 0.05,
    "min_strength_threshold": 0.05,
    "high_retrieval_boost": 2.0
  }
}
```

### Learning-Optimized
```json
{
  "evolution": {
    "exponential_decay_rate": 0.15,
    "spaced_repetition_boost": true,
    "learning_curve_adaptation": true
  }
}
```

## Advanced Features

### Adaptive Decay Rates
- **Content Type**: Different rates for different memory types
- **User Patterns**: Adapt to individual forgetting patterns
- **Domain Sensitivity**: Adjust for subject matter expertise
- **Context Relevance**: Slower decay for currently relevant content

### Memory Rehabilitation
- **Threshold Recovery**: Restore memories just above archive threshold
- **Connection Revival**: Strengthen through related memory access
- **User Intervention**: Manual memory strengthening
- **Context Reactivation**: Strengthen through contextual cues

## Best Practices

1. **Rate Calibration**: Tune decay rates for your use case
2. **Retrieval Tracking**: Monitor access patterns accurately
3. **Quality Metrics**: Measure decay effectiveness
4. **User Feedback**: Incorporate user memory preferences
5. **Regular Evaluation**: Adjust parameters based on results

## Integration Features

### Cross-Memory Support
- **Memory Type Sensitivity**: Different decay for working/episodic/semantic
- **Relationship Preservation**: Maintain connections during decay
- **Quality Correlation**: Link decay to memory quality metrics
- **Usage Context**: Consider memory usage patterns

### User Experience
- **Transparent Decay**: Users understand why memories fade
- **Recovery Options**: Easy restoration of archived memories
- **Strength Indicators**: Show current memory strength
- **Access Recommendations**: Suggest memories needing reinforcement

## Related Evolvers

- [Retrieval-Based Strengthening](./retrieval-strengthening) - Memory access benefits
- [Interference-Based Consolidation](./interference-consolidation) - Memory competition
- [Episodic Decay](./episodic-decay) - Time-based episodic archival
