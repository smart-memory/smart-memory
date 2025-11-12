# Episodic to Zettel Evolution

The **EpisodicToZettelEvolver** converts episodic events into Zettelkasten notes on a periodic basis, transforming experiences into interconnected knowledge notes.

## Overview

This evolver bridges episodic experiences and the Zettelkasten knowledge management system, creating atomic, linkable notes from episodic events that can form part of a larger knowledge graph.

## When Evolution Triggers

### Periodic Rollup
- **Trigger**: Time-based periodic evaluation
- **Default Period**: 1 day (configurable)
- **Frequency**: Daily by default
- **Automatic**: Yes, part of background evolution

### Configuration
```json
{
  "evolution": {
    "episodic_to_zettel_period": 1
  }
}
```

## Evolution Process

1. **Temporal Window**: Collect episodic events from specified period
2. **Event Analysis**: Evaluate events for Zettel conversion suitability
3. **Note Creation**: Transform events into atomic Zettel notes
4. **Auto-Linking**: Connect new Zettels to existing knowledge graph
5. **Metadata Transfer**: Preserve episodic context and timestamps

## Implementation Details

```python
class EpisodicToZettelEvolver(Evolver):
    def evolve(self, memory, logger=None):
        period = self.config.get("episodic_to_zettel_period", 1)  # days
        events = memory.episodic.get_events_since(days=period)
        
        for event in events:
            zettel = memory.zettel.create_note_from_event(event)
            memory.zettel.add(zettel)
```

## Event to Zettel Conversion

### Suitable Events
- **Learning Insights**: New knowledge or understanding
- **Decision Points**: Important choices and their rationale
- **Problem Solutions**: How issues were resolved
- **Pattern Recognition**: Identified trends or connections
- **Key Conversations**: Important discussions or meetings

### Conversion Process
1. **Content Extraction**: Extract key insights from episodic event
2. **Atomicity Check**: Ensure note contains single coherent idea
3. **Title Generation**: Create descriptive title for the Zettel
4. **Tag Assignment**: Add relevant tags from episodic metadata
5. **Link Discovery**: Find connections to existing Zettels

## Zettel Enhancement

### Automatic Features
- **Entity Extraction**: Identify key concepts and entities
- **Semantic Linking**: Connect to related existing notes
- **Temporal Context**: Preserve when the insight occurred
- **Source Tracking**: Link back to original episodic event

### Example Conversion
```
Episodic Event:
"Had breakthrough in debugging session on 2024-01-15. Realized memory leak 
was caused by circular references in event handlers."

Zettel Note:
Title: "Circular References in Event Handlers Cause Memory Leaks"
Content: "Event handlers that maintain references to their parent objects 
can create circular reference chains, preventing garbage collection."
Tags: #debugging #memory-leaks #javascript #event-handlers
Links: [[garbage-collection.md]], [[event-driven-architecture.md]]
```

## Knowledge Graph Integration

### Automatic Linking
- **Semantic Similarity**: Links based on content similarity
- **Entity Overlap**: Connects notes sharing entities
- **Tag Relationships**: Links notes with common tags
- **Temporal Clustering**: Groups notes from similar time periods

### Graph Growth
- **Emergent Structure**: Knowledge organization emerges naturally
- **Cross-Domain Connections**: Links between different topic areas
- **Learning Pathways**: Tracks knowledge development over time
- **Insight Networks**: Maps how insights build upon each other

## Benefits

- **Knowledge Preservation**: Converts fleeting insights into permanent notes
- **Interconnected Learning**: Builds connected knowledge networks
- **Atomic Knowledge**: Creates focused, reusable knowledge units
- **Temporal Tracking**: Maintains connection to when insights occurred

## Configuration Options

### Time-Based Settings
```json
{
  "evolution": {
    "episodic_to_zettel_period": 1,           // days
    "minimum_event_age": 0.5,                 // hours
    "maximum_events_per_period": 50
  }
}
```

### Quality Filters
```json
{
  "evolution": {
    "minimum_insight_score": 0.7,
    "require_unique_content": true,
    "skip_routine_events": true,
    "prioritize_learning_events": true
  }
}
```

### Advanced Features
```json
{
  "evolution": {
    "enable_cross_references": true,
    "auto_generate_summaries": true,
    "create_topic_clusters": true,
    "maintain_episodic_links": true
  }
}
```

## Best Practices

1. **Regular Processing**: Maintain consistent conversion schedules
2. **Quality Over Quantity**: Focus on meaningful insights
3. **Link Maintenance**: Ensure proper knowledge graph connections
4. **Review Process**: Periodically review generated Zettels
5. **Tagging Consistency**: Maintain coherent tagging system

## Common Use Cases

### Research & Learning
- Convert research insights into knowledge notes
- Build interconnected understanding of topics
- Track learning progression over time

### Project Management
- Capture decision rationale as Zettels
- Document problem-solving approaches
- Build institutional knowledge

### Personal Development
- Transform experiences into learnings
- Build personal knowledge management system
- Track growth and insights over time

## Related Evolvers

- [Episodic to Semantic](./episodic-to-semantic) - Episodic to factual knowledge
- [Zettel Pruning](./zettel-pruning) - Zettel quality maintenance
- [Episodic Decay](./episodic-decay) - Episodic memory archival
