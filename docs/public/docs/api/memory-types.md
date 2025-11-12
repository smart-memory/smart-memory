# Memory Types API

SmartMemory supports five distinct memory types, each optimized for different kinds of information and access patterns. This document provides detailed API documentation for working with each memory type.

## Memory Type Overview

| Memory Type | Purpose | Storage Pattern | Access Pattern |
|-------------|---------|-----------------|----------------|
| **Semantic** | Facts, knowledge, concepts | Graph-based with rich relationships | Similarity-based search |
| **Episodic** | Events, experiences, sequences | Temporal ordering with context | Time-based and contextual |
| **Procedural** | Skills, processes, workflows | Step-based structures | Task-oriented retrieval |
| **Working** | Active context, temporary data | Short-term with expiration | Recency-based access |
| **Zettelkasten** | Atomic knowledge, interconnected notes | Knowledge graph with auto-linking | Graph navigation and semantic discovery |

## Semantic Memory API

### Overview
Semantic memory stores factual knowledge, concepts, and their relationships. It's optimized for knowledge representation and semantic search.

### Core Operations

#### Adding Semantic Memories
```python
# Basic semantic memory addition
memory_id = memory.add(
    content="Python is a high-level programming language",
    memory_type="semantic",
    metadata={
        "domain": "programming",
        "language": "python",
        "concepts": ["programming_language", "high_level"]
    }
)

# Structured semantic memory
semantic_item = {
    "content": "Machine learning algorithms learn patterns from data",
    "entities": [
        {"name": "machine_learning", "type": "field"},
        {"name": "algorithms", "type": "method"},
        {"name": "patterns", "type": "concept"},
        {"name": "data", "type": "resources"}
    ],
    "relationships": [
        {"source": "algorithms", "target": "patterns", "type": "LEARNS"},
        {"source": "patterns", "target": "data", "type": "FROM"}
    ]
}

memory_id = memory.add(semantic_item, memory_type="semantic")
```

#### Searching Semantic Memory
```python
# Concept-based search
results = memory.search(
    query="programming languages",
    memory_type="semantic",
    top_k=10,
    filters={
        "domain": "programming",
        "confidence_threshold": 0.7
    }
)

# Entity-based search
entity_results = memory.search_by_entity(
    entity_name="python",
    entity_type="programming_language",
    relationship_types=["IS_A", "USED_FOR"]
)

# Relationship traversal
related_concepts = memory.get_related_concepts(
    concept="machine_learning",
    relationship_types=["IS_A", "PART_OF", "USES"],
    max_depth=2
)
```

### Semantic Memory Methods

#### `SemanticMemory.add_concept(concept, properties=None)`
Add a new concept to semantic memory.

**Parameters:**
- `concept` (str): Concept name or identifier
- `properties` (dict, optional): Concept properties and attributes

**Returns:**
- `str`: Concept ID

**Example:**
```python
concept_id = memory.semantic.add_concept(
    "neural_networks",
    properties={
        "type": "algorithm",
        "complexity": "high",
        "applications": ["nlp", "computer_vision"]
    }
)
```

#### `SemanticMemory.link_concepts(concept1, concept2, relationship_type, strength=1.0)`
Create relationship between concepts.

**Parameters:**
- `concept1` (str): Source concept ID
- `concept2` (str): Target concept ID  
- `relationship_type` (str): Type of relationship
- `strength` (float): Relationship strength (0.0-1.0)

**Returns:**
- `str`: Relationship ID

#### `SemanticMemory.get_concept_hierarchy(root_concept, max_depth=None)`
Retrieve concept hierarchy starting from root.

**Parameters:**
- `root_concept` (str): Root concept ID
- `max_depth` (int, optional): Maximum traversal depth

**Returns:**
- `dict`: Hierarchical concept structure

## Episodic Memory API

### Overview
Episodic memory stores temporal experiences, events, and their contextual information. It maintains chronological ordering and contextual relationships.

### Core Operations

#### Adding Episodic Memories
```python
# Event-based episodic memory
event_memory = {
    "content": "User asked about machine learning algorithms",
    "event_type": "question",
    "timestamp": "2024-01-15T10:30:00Z",
    "context": {
        "user_id": "user123",
        "session_id": "session456",
        "conversation_turn": 5
    },
    "participants": ["user123", "assistant"],
    "location": "chat_interface"
}

memory_id = memory.add(event_memory, memory_type="episodic")

# Sequence-based episodic memory
sequence = [
    {"action": "user_login", "timestamp": "2024-01-15T10:00:00Z"},
    {"action": "browse_docs", "timestamp": "2024-01-15T10:15:00Z"},
    {"action": "ask_question", "timestamp": "2024-01-15T10:30:00Z"},
    {"action": "receive_answer", "timestamp": "2024-01-15T10:31:00Z"}
]

sequence_id = memory.add_sequence(sequence, memory_type="episodic")
```

#### Searching Episodic Memory
```python
# Time-based search
recent_memories = memory.search(
    query="machine learning questions",
    memory_type="episodic",
    time_range={
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-01-31T23:59:59Z"
    }
)

# Context-based search
user_interactions = memory.search(
    memory_type="episodic",
    filters={
        "user_id": "user123",
        "event_type": "question"
    },
    sort_by="timestamp",
    order="desc"
)

# Sequence pattern search
patterns = memory.find_sequence_patterns(
    pattern=["user_login", "ask_question"],
    time_window="1h",
    min_support=0.1
)
```

### Episodic Memory Methods

#### `EpisodicMemory.add_event(event_data, context=None)`
Add a single event to episodic memory.

**Parameters:**
- `event_data` (dict): Event information
- `context` (dict, optional): Additional context

**Returns:**
- `str`: Event memory ID

#### `EpisodicMemory.add_sequence(events, sequence_metadata=None)`
Add a sequence of related events.

**Parameters:**
- `events` (list): List of event dictionaries
- `sequence_metadata` (dict, optional): Sequence-level metadata

**Returns:**
- `str`: Sequence ID

#### `EpisodicMemory.get_timeline(user_id=None, start_time=None, end_time=None)`
Retrieve chronological timeline of events.

**Parameters:**
- `user_id` (str, optional): Filter by user
- `start_time` (datetime, optional): Timeline start
- `end_time` (datetime, optional): Timeline end

**Returns:**
- `list`: Chronologically ordered events

#### `EpisodicMemory.find_similar_episodes(reference_event, similarity_threshold=0.7)`
Find episodes similar to reference event.

**Parameters:**
- `reference_event` (str): Reference event ID
- `similarity_threshold` (float): Minimum similarity score

**Returns:**
- `list`: Similar episodes with similarity scores

## Procedural Memory API

### Overview
Procedural memory stores skills, processes, and step-by-step procedures. It's optimized for task-oriented knowledge and workflow representation.

### Core Operations

#### Adding Procedural Memories
```python
# Skill-based procedural memory
skill = {
    "name": "data_preprocessing",
    "description": "Steps for preprocessing machine learning data",
    "steps": [
        {
            "step_id": 1,
            "action": "load_data",
            "description": "Load raw data from source",
            "inputs": ["data_source"],
            "outputs": ["raw_data"],
            "tools": ["pandas", "numpy"]
        },
        {
            "step_id": 2,
            "action": "clean_data",
            "description": "Remove missing values and outliers",
            "inputs": ["raw_data"],
            "outputs": ["clean_data"],
            "dependencies": [1]
        }
    ],
    "prerequisites": ["python_basics", "pandas_knowledge"],
    "difficulty": "intermediate"
}

skill_id = memory.add(skill, memory_type="procedural")

# Workflow-based procedural memory
workflow = {
    "name": "ml_model_training",
    "type": "workflow",
    "stages": [
        {"name": "data_preparation", "procedures": ["data_preprocessing"]},
        {"name": "model_selection", "procedures": ["algorithm_selection"]},
        {"name": "training", "procedures": ["model_training"]},
        {"name": "evaluation", "procedures": ["model_evaluation"]}
    ]
}

workflow_id = memory.add(workflow, memory_type="procedural")
```

#### Searching Procedural Memory
```python
# Skill-based search
skills = memory.search(
    query="data preprocessing techniques",
    memory_type="procedural",
    filters={
        "skill_type": "data_science",
        "difficulty": ["beginner", "intermediate"]
    }
)

# Task-oriented search
procedures = memory.search_procedures(
    task="clean messy data",
    domain="machine_learning",
    available_tools=["pandas", "numpy", "scikit-learn"]
)

# Workflow search
workflows = memory.get_workflows_for_goal(
    goal="train classification models",
    constraints={"time_limit": "2h", "experience_level": "intermediate"}
)
```

### Procedural Memory Methods

#### `ProceduralMemory.add_skill(skill_data)`
Add a new skill to procedural memory.

**Parameters:**
- `skill_data` (dict): Skill definition with steps and metadata

**Returns:**
- `str`: Skill ID

#### `ProceduralMemory.add_workflow(workflow_data)`
Add a workflow composed of multiple procedures.

**Parameters:**
- `workflow_data` (dict): Workflow definition

**Returns:**
- `str`: Workflow ID

#### `ProceduralMemory.get_execution_plan(goal, constraints=None)`
Generate execution plan for achieving a goal.

**Parameters:**
- `goal` (str): Desired outcome
- `constraints` (dict, optional): Execution constraints

**Returns:**
- `dict`: Detailed execution plan with steps and dependencies

#### `ProceduralMemory.optimize_procedure(procedure_id, optimization_criteria)`
Optimize existing procedure based on criteria.

**Parameters:**
- `procedure_id` (str): Procedure to optimize
- `optimization_criteria` (dict): Optimization parameters

**Returns:**
- `dict`: Optimized procedure definition

## Working Memory API

### Overview
Working memory manages active, short-term information with automatic expiration and capacity limits. It's optimized for current context and temporary data.

### Core Operations

#### Adding Working Memory
```python
# Context-based working memory
context = {
    "current_task": "answering_ml_question",
    "active_concepts": ["neural_networks", "deep_learning"],
    "user_preferences": {"detail_level": "intermediate"},
    "conversation_state": {
        "turn": 5,
        "topic": "machine_learning",
        "last_question": "What are CNNs?"
    }
}

context_id = memory.add(context, memory_type="working", ttl=3600)  # 1 hour TTL

# Temporary data
temp_data = {
    "search_results": [{"id": "mem_123", "score": 0.95}],
    "processing_state": "analyzing_query",
    "intermediate_results": {"entities": ["CNN", "neural_network"]}
}

temp_id = memory.add(temp_data, memory_type="working", ttl=300)  # 5 minutes TTL
```

#### Accessing Working Memory
```python
# Get current context
current_context = memory.get_working_context(
    context_type="conversation",
    user_id="user123"
)

# Get active items
active_items = memory.get_active_working_memory(
    max_age_seconds=3600,
    sort_by="access_time"
)

# Update working memory
memory.update_working_memory(
    context_id,
    updates={"conversation_state.turn": 6}
)
```

### Working Memory Methods

#### `WorkingMemory.add_context(context_data, ttl=3600)`
Add contextual information to working memory.

**Parameters:**
- `context_data` (dict): Context information
- `ttl` (int): Time-to-live in seconds

**Returns:**
- `str`: Context ID

#### `WorkingMemory.get_active_context(context_type=None, user_id=None)`
Retrieve currently active context.

**Parameters:**
- `context_type` (str, optional): Type of context
- `user_id` (str, optional): User identifier

**Returns:**
- `dict`: Active context data

#### `WorkingMemory.update_context(context_id, updates)`
Update existing working memory context.

**Parameters:**
- `context_id` (str): Context identifier
- `updates` (dict): Updates to apply

**Returns:**
- `bool`: Success status

#### `WorkingMemory.cleanup_expired()`
Remove expired working memory items.

**Returns:**
- `dict`: Cleanup statistics

## Cross-Memory Type Operations

### Memory Type Interactions

#### Promoting Working Memory to Long-term
```python
# Promote important working memory to episodic
important_context = memory.get_working_memory(context_id)
if important_context["importance"] > 0.8:
    episodic_id = memory.promote_to_episodic(
        context_id,
        event_type="important_context"
    )

# Convert procedural steps to semantic knowledge
procedure = memory.get_procedure(procedure_id)
for step in procedure["steps"]:
    if step["generalizable"]:
        semantic_id = memory.add(
            content=step["knowledge"],
            memory_type="semantic",
            source_procedure=procedure_id
        )
```

#### Cross-Type Search
```python
# Search across multiple memory types
results = memory.search_multi_type(
    query="machine learning best practices",
    memory_types=["semantic", "procedural", "episodic"],
    merge_strategy="weighted",
    type_weights={
        "semantic": 0.4,
        "procedural": 0.4,
        "episodic": 0.2
    }
)

# Find related memories across types
related = memory.find_cross_type_relationships(
    anchor_memory_id="semantic_123",
    relationship_types=["IMPLEMENTS", "EXEMPLIFIES", "USES"],
    target_types=["procedural", "episodic"]
)
```

## Memory Type Configuration

### Type-Specific Settings

```python
memory_type_config = {
    "semantic": {
        "enable_concept_extraction": True,
        "relationship_threshold": 0.6,
        "max_relationships_per_item": 20,
        "enable_ontology_integration": True
    },
    "episodic": {
        "enable_temporal_clustering": True,
        "sequence_detection": True,
        "context_window_size": 10,
        "temporal_decay_factor": 0.95
    },
    "procedural": {
        "enable_step_optimization": True,
        "dependency_analysis": True,
        "skill_level_inference": True,
        "workflow_composition": True
    },
    "working": {
        "default_ttl": 3600,
        "max_capacity": 1000,
        "cleanup_interval": 300,
        "importance_threshold": 0.5
    }
}

memory = SmartMemory(memory_type_config=memory_type_config)
```

## Best Practices

### Memory Type Selection

1. **Semantic Memory**: Use for facts, concepts, and knowledge that should persist
2. **Episodic Memory**: Use for events, experiences, and temporal sequences
3. **Procedural Memory**: Use for skills, processes, and step-by-step procedures
4. **Working Memory**: Use for temporary context and active processing state

### Performance Optimization

1. **Type-Specific Indexing**: Each memory type uses optimized indexes
2. **Targeted Search**: Search specific memory types when possible
3. **Appropriate TTL**: Set reasonable TTL for working memory
4. **Batch Operations**: Use batch operations for multiple items

### Memory Lifecycle

1. **Working → Episodic**: Promote important temporary context
2. **Episodic → Semantic**: Extract generalizable knowledge from experiences
3. **Procedural ↔ Semantic**: Link procedures with conceptual knowledge
4. **Cross-Type Relationships**: Maintain relationships between memory types

## Next Steps

- **Advanced Features**: Explore [advanced features](../guides/advanced-features.md) for custom memory type behaviors
- **Performance Tuning**: Optimize memory type operations with [performance tuning](../guides/performance-tuning.md)
- **Core API**: Complete SmartMemory API in [SmartMemory API](smart-memory.md)
- **Examples**: See memory types in action in [conversational AI examples](../examples/conversational-ai.md)

Each memory type provides specialized capabilities optimized for different kinds of information and access patterns. Understanding when and how to use each type is key to building effective agentic memory systems.
