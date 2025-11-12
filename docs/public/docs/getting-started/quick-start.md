# Quick Start


Get up and running with SmartMemory in minutes. This guide will walk you through creating your first memory-enabled application with **multi-user support**.

Before using SmartMemory, ensure the backend services are running:

```bash
# Start all required services
docker-compose up -d
docker-compose ps
```
## Basic Usage

### 1. Initialize SmartMemory

```python
from smartmemory import SmartMemory

# Initialize with default configuration
memory = SmartMemory()

# Or with custom configuration
# memory = SmartMemory(config_path="config.json")
```
### 2. Add Memories

SmartMemory automatically processes and enriches memories:

```python
# Add memories for different users (user isolation via user_id parameter)
memory.add("I learned Python programming in 2020", user_id="alice")
memory.add("Paris is the capital of France", user_id="bob")
memory.add("To make coffee: heat water, add grounds, brew for 4 minutes", user_id="alice")
memory.add("Yesterday I had lunch with Sarah at the Italian restaurant", user_id="charlie")
```
### 3. Search Memories with User Filtering

Search for relevant memories for specific users:
```python
# Search for relevant memories for specific users
alice_results = memory.search("programming", user_id="alice")
bob_results = memory.search("France", user_id="bob")

# Each user only sees their own memories
for result in alice_results:
    print(f"Alice's memory: {result.content}")

for result in bob_results:
    print(f"Bob's memory: {result.content}")
```

## Multi-User Features

### User Isolation

SmartMemory provides **enterprise-grade user isolation**:

```python
# Users only access their own memories
user_a_memories = memory.search("coffee", user_id="user_a")  # Only user_a's coffee memories
user_b_memories = memory.search("coffee", user_id="user_b")  # Only user_b's coffee memories

# No cross-user contamination
assert len(user_a_memories) != len(user_b_memories)  # Different results for different users
```

## Complete Example: Personal Assistant with Multi-User Support

```python
from smartmemory import SmartMemory
import datetime
class PersonalAssistant:
    def __init__(self):
        self.memory = SmartMemory()
        
    def learn(self, information, user_id):
        """Add new information to memory"""
        # Add timestamp for episodic memories
        timestamped_info = f"[{datetime.datetime.now()}] {information}"
        result = self.memory.add(timestamped_info, user_id=user_id)
        print(f"Learned: {information}")
        return result
        
    def recall(self, query, user_id):
        """Search for relevant memories"""
        results = self.memory.search(query, user_id=user_id, top_k=3)
        if results:
            print(f"I remember {len(results)} things about '{query}':")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.content}")
        else:
            print(f"I don't remember anything about '{query}'")
        return results
        
    def get_context(self, topic, user_id):
        """Get comprehensive context about a topic"""
        # Search across all memory types
        all_results = []
        for memory_type in ["semantic", "episodic", "procedural"]:
            results = self.memory.search(topic, user_id=user_id, memory_type=memory_type, top_k=2)
            all_results.extend(results)
        
        return all_results

# Usage example
assistant = PersonalAssistant()

# Learn various things for different users
assistant.learn("My favorite coffee shop is Blue Bottle", user_id="alice")
assistant.learn("I have a meeting with John tomorrow at 2 PM", user_id="bob")
assistant.learn("To reset WiFi router: unplug for 30 seconds, then plug back in", user_id="alice")
assistant.learn("Python is a programming language I use for data analysis", user_id="charlie")

# Recall information for specific users
assistant.recall("coffee", user_id="alice")
assistant.recall("meeting", user_id="bob")
assistant.recall("Python", user_id="charlie")

# Get comprehensive context for a topic
context = assistant.get_context("programming", user_id="alice")
print(f"Programming context: {len(context)} memories")
```

## Working with Different Memory Types

### Semantic Memory (Facts & Knowledge)

```python
# Add factual information with user isolation
memory.add("The speed of light is 299,792,458 meters per second", user_id="alice")
memory.add("Machine learning is a subset of artificial intelligence", user_id="bob")
memory.add("Tokyo is the capital of Japan", user_id="alice")

# Search semantic knowledge (user_id parameter filters results)
facts = memory.search("artificial intelligence", user_id="bob", memory_type="semantic")
```

### Episodic Memory (Personal Experiences)

```python
# Add personal experiences with context
memory.add("I attended the AI conference in San Francisco last week", user_id="alice")
memory.add("Had dinner at the new sushi place downtown yesterday", user_id="bob")
memory.add("Completed the machine learning course on Coursera in March", user_id="alice")

# Search personal experiences
experiences = memory.search("conference", user_id="alice", memory_type="episodic")
```

### Procedural Memory (How-to Knowledge)

```python
# Add step-by-step procedures
memory.add("To deploy to AWS: 1) Build Docker image 2) Push to ECR 3) Update ECS service", user_id="alice")
memory.add("Git workflow: 1) Create branch 2) Make changes 3) Commit 4) Push 5) Create PR", user_id="bob")
memory.add("Morning routine: 1) Exercise 2) Shower 3) Coffee 4) Check emails", user_id="alice")

# Search procedures
procedures = memory.search("deploy", user_id="alice", memory_type="procedural")
```

## Background Processing

Enable background processing for better performance:

```python
# Synchronous (process now via full pipeline)
memory.ingest("Process immediately", user_id="alice", sync=True)

# Asynchronous (quick persist + enqueue for background workers)
result = memory.ingest("Process in background", user_id="alice", sync=False)
print(result)  # {"item_id": "...", "queued": True}

# Note: a separate worker service must consume the background queue.
```

## Advanced Features

### Custom Linking

```python
# Create explicit relationships between memories
python_id = memory.add("Python programming language", user_id="alice")
ds_id = memory.add("Data science projects", user_id="alice")

# Link related concepts
memory.link(
    source_id=python_id,
    target_id=ds_id,
    link_type="USED_FOR"
)
```

### Memory Evolution

```python
# Trigger memory evolution (consolidation, pruning, enhancement)
memory.run_evolution_cycle()
```

### Ontology Management

```python
from smartmemory.ontology import OntologyManager

# Load or create ontology
ontology_manager = OntologyManager()
ontology = ontology_manager.get_active_ontology()

# Add structured knowledge
memory.add({
    "content": "John Smith works at Google",
    "entities": [
        {"name": "John Smith", "type": "Person"},
        {"name": "Google", "type": "Organization"}
    ],
    "relations": [
        {"source": "John Smith", "target": "Google", "type": "WORKS_AT"}
    ]
}, user_id="alice")
```

## Integration Examples

### With LangChain

```python
from langchain.memory import ConversationBufferMemory
from smartmemory import SmartMemory

class SmartMemoryLangChain:
    def __init__(self):
        self.smart_memory = SmartMemory()
        self.conversation_memory = ConversationBufferMemory()
    
    def add_conversation(self, human_input, ai_response, user_id):
        # Store in both systems
        self.conversation_memory.save_context(
            {"input": human_input},
            {"output": ai_response}
        )
        
        # Add to SmartMemory for long-term retention
        self.smart_memory.add(f"User asked: {human_input}", user_id=user_id)
        self.smart_memory.add(f"I responded: {ai_response}", user_id=user_id)
```

### With OpenAI API

```python
import openai
from smartmemory import SmartMemory

class MemoryEnhancedChatbot:
    def __init__(self):
        self.memory = SmartMemory()
        
    def chat(self, user_message, user_id):
        # Search for relevant context
        context = self.memory.search(user_message, user_id=user_id, top_k=3)
        context_text = "\n".join([c.content for c in context])
        
        # Create prompt with memory context
        prompt = f"""
        Previous context:
        {context_text}
        
        User: {user_message}
        Assistant:"""
        
        # Get AI response
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        ai_response = response.choices[0].message.content
        
        # Store conversation in memory
        self.memory.add(f"User said: {user_message}", user_id=user_id)
        self.memory.add(f"I responded: {ai_response}", user_id=user_id)
        
        return ai_response
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize SmartMemory behavior
- [Core Concepts](../concepts/overview.md) - Understand the architecture
- [API Reference](../api/smart-memory.md) - Detailed API documentation
- [Advanced Features](../guides/advanced-features.md) - Explore powerful capabilities
