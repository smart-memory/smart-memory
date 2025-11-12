# Working to Procedural Evolution

The **WorkingToProceduralEvolver** identifies repeated skill and tool usage patterns in working memory and promotes them to procedural memory as reusable macros.

## Overview

This evolver implements skill acquisition by detecting patterns of repeated tool usage or procedures in working memory and converting them into automated procedural knowledge, similar to how humans develop muscle memory and automatic responses.

## When Evolution Triggers

### Pattern Detection
- **Trigger**: Repeated skill patterns exceed threshold
- **Default Threshold**: 5 occurrences (configurable via `working_to_procedural_k`)
- **Frequency**: Continuous monitoring during evolution cycles
- **Automatic**: Yes, when patterns are detected

### Configuration
```json
{
  "evolution": {
    "working_to_procedural_k": 5,
    "pattern_detection": {
      "enable_tool_patterns": true,
      "enable_sequence_patterns": true,
      "minimum_pattern_length": 2,
      "similarity_threshold": 0.8
    }
  }
}
```

## Evolution Process

1. **Pattern Analysis**: Scan working memory for repeated sequences
2. **Skill Detection**: Identify tool usage and procedure patterns
3. **Frequency Counting**: Track occurrence of similar patterns
4. **Threshold Evaluation**: Check if patterns meet promotion criteria
5. **Macro Creation**: Convert patterns into procedural macros
6. **Automation Setup**: Enable automatic execution of procedures

## Implementation Details

```python
class WorkingToProceduralEvolver(Evolver):
    def evolve(self, memory, logger=None):
        k = self.config.get("working_to_procedural_k", 5)
        patterns = memory.working.detect_skill_patterns(min_count=k)
        
        for pattern in patterns:
            memory.procedural.add_macro(pattern)
            if logger:
                logger.info(f"Promoted working skill pattern to procedural: {pattern}")
```

## Pattern Types

### Tool Usage Patterns
- **Command Sequences**: Repeated sequences of tool invocations
- **Parameter Patterns**: Common parameter combinations
- **Workflow Steps**: Multi-step procedures involving tools
- **Context Switches**: Patterns of switching between tools

### Skill Patterns
- **Problem-Solving**: Repeated approaches to similar problems
- **Debug Procedures**: Common debugging sequences
- **Code Patterns**: Frequently used code structures
- **Search Strategies**: Effective information retrieval patterns

## Example Patterns

### Development Workflow
```
Working Memory Pattern:
1. git status
2. git add .
3. git commit -m "message"
4. git push origin main

Procedural Macro:
"commit_and_push" → Automated sequence with parameter for commit message
```

### Debugging Pattern
```
Working Memory Pattern:
1. Check logs for errors
2. Reproduce issue locally
3. Add debug statements
4. Test fix
5. Clean up debug code

Procedural Macro:
"debug_workflow" → Guided debugging procedure with checkpoints
```

### Research Pattern
```
Working Memory Pattern:
1. Search for concept
2. Read documentation
3. Test simple example
4. Apply to current context

Procedural Macro:
"research_and_apply" → Learning workflow with context awareness
```

## Macro Features

### Automatic Execution
- **Trigger Recognition**: Detect when macro should activate
- **Parameter Binding**: Automatically fill common parameters
- **Context Awareness**: Adapt execution based on current state
- **Error Handling**: Manage failures and edge cases

### Customization
- **Parameter Slots**: Configurable inputs for flexible execution
- **Conditional Steps**: Skip or modify steps based on conditions
- **User Overrides**: Allow manual intervention when needed
- **Learning Updates**: Improve macro based on usage patterns

## Benefits

- **Efficiency Gains**: Automate repetitive procedures
- **Cognitive Load Reduction**: Free working memory for complex tasks
- **Consistency**: Ensure procedures are followed reliably
- **Skill Transfer**: Codify expertise into reusable patterns

## Advanced Features

### Pattern Recognition
- **Semantic Similarity**: Detect functionally similar sequences
- **Temporal Clustering**: Group patterns by time of occurrence
- **Context Filtering**: Focus on patterns relevant to current domain
- **Cross-Session Learning**: Learn patterns across multiple sessions

### Macro Management
- **Usage Tracking**: Monitor how often macros are invoked
- **Performance Metrics**: Measure macro effectiveness
- **Automatic Updates**: Refine macros based on new patterns
- **Conflict Resolution**: Handle overlapping or competing macros

## Configuration Examples

### Conservative Learning
```json
{
  "evolution": {
    "working_to_procedural_k": 10,
    "require_exact_matches": true,
    "manual_approval_required": true,
    "pattern_confidence_threshold": 0.9
  }
}
```

### Aggressive Automation
```json
{
  "evolution": {
    "working_to_procedural_k": 3,
    "enable_fuzzy_matching": true,
    "auto_create_macros": true,
    "pattern_confidence_threshold": 0.7
  }
}
```

### Domain-Specific Settings
```json
{
  "evolution": {
    "coding_patterns": {
      "working_to_procedural_k": 3,
      "focus_on_tool_sequences": true
    },
    "research_patterns": {
      "working_to_procedural_k": 5,
      "include_search_patterns": true
    }
  }
}
```

## Best Practices

1. **Pattern Quality**: Ensure patterns are genuinely useful
2. **User Control**: Provide oversight of macro creation
3. **Context Sensitivity**: Consider when patterns should activate
4. **Performance Monitoring**: Track macro effectiveness
5. **Regular Review**: Periodically audit and update macros

## Integration with Tools

### Tool Pattern Detection
- **API Call Sequences**: Repeated API usage patterns
- **File Operations**: Common file manipulation sequences
- **System Commands**: Frequently used command combinations
- **Search Queries**: Effective search strategies

### Macro Execution
- **Tool Integration**: Seamless integration with existing tools
- **Parameter Passing**: Automatic parameter binding
- **Error Recovery**: Robust error handling and recovery
- **Progress Tracking**: Monitor macro execution progress

## Related Evolvers

- [Working to Episodic](./working-to-episodic) - Working memory consolidation
- [Rapid Enrichment](./rapid-enrichment) - Immediate knowledge enhancement
- [Strategic Pruning](./strategic-pruning) - Procedural memory cleanup
