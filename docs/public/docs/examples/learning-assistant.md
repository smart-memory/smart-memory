# Learning Assistant Example

This example demonstrates how to build an intelligent learning assistant using SmartMemory that can track learning progress, provide personalized recommendations, and adapt to individual learning patterns.

## Overview

The Learning Assistant leverages SmartMemory's multiple memory types to create a comprehensive learning experience:

- **Semantic Memory**: Store facts, concepts, and knowledge
- **Episodic Memory**: Track learning sessions and experiences
- **Procedural Memory**: Remember learning strategies and methods
- **Working Memory**: Maintain current learning context

## Implementation

### Basic Setup

```python
from smartmemory import SmartMemory
from datetime import datetime, timedelta

class LearningAssistant:
    def __init__(self):
        self.memory = SmartMemory(
            config={
                "evolution": {
                    "consolidation": {"enabled": True},
                    "relationship_discovery": {"enabled": True}
                }
            }
        )
        self.current_session = None
    
    def start_learning_session(self, topic, learning_goals=None):
        """Start a new learning session"""
        session_data = {
            "topic": topic,
            "start_time": datetime.now().isoformat(),
            "goals": learning_goals or [],
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # Store session start in episodic memory
        self.memory.ingest({
            "content": f"Started learning session on {topic}",
            "memory_type": "episodic",
            "metadata": session_data
        })
        
        self.current_session = session_data
        return session_data["session_id"]
```

### Knowledge Tracking

```python
import smartmemory.utils


def learn_concept(self, concept, definition, examples=None):
    """Add a new concept to semantic memory"""
    concept_data = {
        "concept": concept,
        "definition": definition,
        "examples": examples or [],
        "learned_at": smartmemory.utils.now().isoformat(),
        "session_id": self.current_session["session_id"] if self.current_session else None
    }

    # Store in semantic memory
    memory_id = self.memory.ingest({
        "content": f"{concept}: {definition}",
        "memory_type": "semantic",
        "metadata": concept_data
    })

    return memory_id
```

### Usage Example

```python
# Initialize the learning assistant
assistant = LearningAssistant()

# Start a learning session
session_id = assistant.start_learning_session(
    topic="Machine Learning Fundamentals",
    learning_goals=["Understand supervised learning", "Learn about neural networks"]
)

# Learn new concepts
assistant.learn_concept(
    concept="Supervised Learning",
    definition="A type of machine learning where the algorithm learns from labeled training data",
    examples=["Classification", "Regression"]
)

# Get learning progress
progress = assistant.get_learning_progress(topic="Machine Learning", days=7)
print(f"Concepts learned: {progress['concepts_learned']}")
```

## Features

- **Progress Tracking**: Monitor learning over time
- **Knowledge Gaps**: Identify areas needing attention  
- **Personalized Recommendations**: Suggest optimal learning strategies
- **Session Management**: Track individual learning sessions
- **Concept Relationships**: Discover connections between topics

This example shows how SmartMemory's multi-type memory system can create sophisticated learning applications that adapt to individual needs and learning patterns.
