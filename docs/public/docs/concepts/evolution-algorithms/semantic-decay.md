# Semantic Decay Evolution

The **SemanticDecayEvolver** manages the archival and pruning of semantic facts based on low relevance, age, or negative feedback to maintain high-quality semantic knowledge.

## Overview

This evolver implements quality control for semantic memory by identifying and archiving facts that have become irrelevant, outdated, or of low quality, ensuring the semantic knowledge base remains accurate and useful.

## When Evolution Triggers

### Relevance-Based Pruning
- **Trigger**: Semantic facts fall below relevance threshold
- **Default Threshold**: 0.2 (20% relevance)
- **Frequency**: Periodic evaluation during evolution cycles
- **Automatic**: Yes, part of maintenance routines

### Configuration
```json
{
  "evolution": {
    "semantic_decay_threshold": 0.2,
    "include_age_factor": true,
    "require_usage_history": true,
    "preserve_core_knowledge": true
  }
}
```

## Evolution Process

1. **Relevance Analysis**: Evaluate semantic facts for current relevance
2. **Quality Assessment**: Check fact accuracy and usefulness
3. **Usage Pattern Review**: Analyze access and reference patterns
4. **Age Consideration**: Factor in fact staleness and currency
5. **Archival Decision**: Archive or update low-relevance facts

## Implementation Details

```python
class SemanticDecayEvolver(Evolver):
    def evolve(self, memory, logger=None):
        threshold = self.config.get("semantic_decay_threshold", 0.2)
        old_facts = memory.semantic.get_low_relevance(threshold=threshold)
        
        for fact in old_facts:
            memory.semantic.archive(fact)
            if logger:
                logger.info(f"Archived low-relevance semantic fact: {fact}")
```

## Relevance Factors

### Usage-Based Relevance
- **Access Frequency**: How often fact is retrieved
- **Reference Count**: Number of connections to other memories
- **Search Appearances**: Frequency in search results
- **User Engagement**: Explicit user interactions with fact

### Content-Based Relevance
- **Accuracy Score**: Factual correctness assessment
- **Currency**: How up-to-date the information is
- **Uniqueness**: Whether fact provides unique information
- **Comprehensiveness**: Completeness of information

### Context-Based Relevance
- **Domain Applicability**: Relevance to current usage domains
- **Temporal Relevance**: Applicability to current time period
- **User Context**: Relevance to user's current interests
- **Cross-References**: Integration with other knowledge

## Examples

### Outdated Technology Facts
```
Low Relevance: "Internet Explorer 6 supports CSS2 partially"
→ Archived (outdated technology, rarely referenced)
```

### Core Knowledge (Preserved)
```
High Relevance: "Python is an interpreted programming language"
→ Preserved (fundamental fact, frequently referenced)
```

### Context-Dependent Facts
```
Variable Relevance: "Remote work policies updated March 2020"
→ May be archived based on current organizational context
```

## Quality Assessment

### Accuracy Validation
- **Fact Checking**: Automated accuracy assessment
- **Source Verification**: Reliability of information sources
- **Consistency Check**: Alignment with other semantic facts
- **User Corrections**: Incorporation of user feedback

### Utility Measurement
- **Problem Solving**: Usefulness in answering queries
- **Knowledge Building**: Contribution to larger understanding
- **Decision Support**: Value in decision-making processes
- **Learning Aid**: Educational value for users

## Preservation Strategies

### Core Knowledge Protection
- **Fundamental Facts**: Basic domain knowledge preserved
- **High-Confidence Facts**: Well-validated information
- **Frequently Accessed**: Regularly used information
- **User-Marked**: Explicitly important to users

### Conditional Preservation
- **Historical Significance**: Important for context understanding
- **Rare Information**: Unique or hard-to-replace facts
- **Linked Knowledge**: Facts supporting other memories
- **Learning Progress**: Facts part of learning pathways

## Benefits

- **Knowledge Quality**: Maintains high-quality semantic base
- **Performance Optimization**: Faster retrieval of relevant facts
- **Accuracy Improvement**: Removes outdated or incorrect information
- **Focus Enhancement**: Emphasizes currently relevant knowledge

## Configuration Examples

### Conservative Decay
```json
{
  "evolution": {
    "semantic_decay_threshold": 0.1,
    "preserve_all_core_knowledge": true,
    "require_multiple_indicators": true,
    "enable_user_review": true
  }
}
```

### Aggressive Cleanup
```json
{
  "evolution": {
    "semantic_decay_threshold": 0.4,
    "fast_decay_for_outdated": true,
    "prioritize_recent_facts": true,
    "auto_archive_duplicates": true
  }
}
```

### Domain-Specific Settings
```json
{
  "evolution": {
    "technology_facts": {
      "semantic_decay_threshold": 0.3,
      "rapid_obsolescence": true
    },
    "scientific_facts": {
      "semantic_decay_threshold": 0.1,
      "high_preservation": true
    }
  }
}
```

## Advanced Features

### Intelligent Archival
- **Fact Merging**: Combine related low-relevance facts
- **Abstraction**: Convert specific facts to general principles
- **Cross-Reference**: Maintain connections during archival
- **Recovery Mechanisms**: Enable fact restoration when needed

### Dynamic Thresholds
- **Context-Sensitive**: Adjust thresholds based on usage context
- **User-Adaptive**: Personalize based on user patterns
- **Domain-Specific**: Different standards for different knowledge areas
- **Temporal Adjustment**: Modify thresholds over time

## Best Practices

1. **Gradual Implementation**: Start with high thresholds and lower gradually
2. **User Involvement**: Provide user oversight of archival decisions
3. **Recovery Systems**: Maintain ability to restore archived facts
4. **Quality Metrics**: Establish clear relevance measurement criteria
5. **Regular Review**: Periodically audit archival decisions

## Integration Features

### Cross-Memory Coordination
- **Episodic Links**: Consider episodic events supporting facts
- **Procedural Usage**: Preserve facts used in procedures
- **Zettel References**: Maintain facts referenced by Zettels
- **Working Memory**: Respect current working memory needs

### User Control
- **Manual Override**: Users can protect specific facts
- **Relevance Adjustment**: Users can modify fact importance
- **Archive Review**: Users can review archived facts
- **Restoration Tools**: Easy recovery of mistakenly archived facts

## Related Evolvers

- [Episodic Decay](./episodic-decay) - Episodic memory archival
- [Episodic to Semantic](./episodic-to-semantic) - Fact promotion
- [Strategic Pruning](./strategic-pruning) - Cross-memory pruning
