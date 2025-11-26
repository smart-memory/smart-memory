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
### 2. Ingest Memories

SmartMemory automatically processes and enriches memories through the full pipeline:

```python
# Ingest memories (full 11-stage pipeline)
# Input → Classification → Extraction → Storage → Linking → 
# Vector → Enrichment → Grounding → Evolution → Clustering → Versioning
memory.ingest("I learned Python programming in 2020")
memory.ingest("Paris is the capital of France")
memory.ingest("John works at Google as an engineer")  # Entities extracted & clustered
memory.ingest("Yesterday I had lunch with Sarah at the Italian restaurant")

# Or use add() for simple storage without pipeline
from smartmemory import MemoryItem
item = MemoryItem(content="Quick note", memory_type="semantic")
memory.add(item)
```
### 3. Search Memories

Search for relevant memories:
```python
# Search for relevant memories
results = memory.search("programming")
for result in results:
    print(f"Found: {result.content}")

# Filter by memory type
semantic_results = memory.search("France", memory_type="semantic")
episodic_results = memory.search("lunch", memory_type="episodic")
```

**Note:** In multi-tenant deployments (smart-memory-service), user/tenant filtering is handled automatically by `ScopeProvider`.

## Multi-Tenant Features

### Automatic User Isolation

SmartMemory provides **enterprise-grade tenant isolation** via `ScopeProvider`:

```python
# OSS usage (single user, no isolation needed)
memory = SmartMemory()
memory.ingest("My data")  # No scoping

# Service layer (automatic tenant isolation)
# In smart-memory-service, ScopeProvider is injected automatically
from service_common.security import create_secure_smart_memory
memory = create_secure_smart_memory(user, request_scope=scope)
memory.ingest("User data")  # Automatically scoped to tenant
memory.search("query")      # Automatically filtered by tenant
```

## Complete Example: Personal Assistant

```python
from smartmemory import SmartMemory
import datetime

class PersonalAssistant:
    def __init__(self):
        self.memory = SmartMemory()
        
    def learn(self, information):
        """Ingest new information to memory (full pipeline)"""
        timestamped_info = f"[{datetime.datetime.now()}] {information}"
        item_id = self.memory.ingest(timestamped_info)
        print(f"Learned: {information}")
        return item_id
        
    def recall(self, query):
        """Search for relevant memories"""
        results = self.memory.search(query, top_k=3)
        if results:
            print(f"I remember {len(results)} things about '{query}':")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.content}")
        else:
            print(f"I don't remember anything about '{query}'")
        return results
        
    def get_context(self, topic):
        """Get comprehensive context about a topic"""
        all_results = []
        for memory_type in ["semantic", "episodic", "procedural"]:
            results = self.memory.search(topic, memory_type=memory_type, top_k=2)
            all_results.extend(results)
        return all_results

# Usage example
assistant = PersonalAssistant()

# Learn various things
assistant.learn("My favorite coffee shop is Blue Bottle")
assistant.learn("I have a meeting with John tomorrow at 2 PM")
assistant.learn("To reset WiFi router: unplug for 30 seconds, then plug back in")
assistant.learn("Python is a programming language I use for data analysis")

# Recall information
assistant.recall("coffee")
assistant.recall("meeting")
assistant.recall("Python")

# Get comprehensive context
context = assistant.get_context("programming")
print(f"Programming context: {len(context)} memories")
```

## Working with Different Memory Types

### Semantic Memory (Facts & Knowledge)

```python
# Ingest factual information
memory.ingest("The speed of light is 299,792,458 meters per second")
memory.ingest("Machine learning is a subset of artificial intelligence")
memory.ingest("Tokyo is the capital of Japan")

# Search semantic knowledge
facts = memory.search("artificial intelligence", memory_type="semantic")
```

### Episodic Memory (Personal Experiences)

```python
# Ingest personal experiences
memory.ingest("I attended the AI conference in San Francisco last week")
memory.ingest("Had dinner at the new sushi place downtown yesterday")
memory.ingest("Completed the machine learning course on Coursera in March")

# Search personal experiences
experiences = memory.search("conference", memory_type="episodic")
```

### Procedural Memory (How-to Knowledge)

```python
# Ingest step-by-step procedures
memory.ingest("To deploy to AWS: 1) Build Docker image 2) Push to ECR 3) Update ECS service")
memory.ingest("Git workflow: 1) Create branch 2) Make changes 3) Commit 4) Push 5) Create PR")
memory.ingest("Morning routine: 1) Exercise 2) Shower 3) Coffee 4) Check emails")

# Search procedures
procedures = memory.search("deploy", memory_type="procedural")
```

## Background Processing

Enable background processing for better performance:

```python
# Synchronous (process now via full pipeline)
memory.ingest("Process immediately", sync=True)

# Asynchronous (quick persist + enqueue for background workers)
result = memory.ingest("Process in background", sync=False)
print(result)  # {"item_id": "...", "queued": True}

# Note: a separate worker service must consume the background queue.
```

## Advanced Features

### Custom Linking

```python
# Create explicit relationships between memories
python_id = memory.ingest("Python programming language")
ds_id = memory.ingest("Data science projects")

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

# Promote working memory to episodic/procedural
memory.commit_working_to_episodic()
memory.commit_working_to_procedural()
```

### Entity Clustering

```python
# Run clustering to deduplicate entities
stats = memory.run_clustering()
print(f"Merged {stats.get('merged_count', 0)} duplicate entities")

# Clustering uses:
# 1. SemHash pre-deduplication (fast, 0.95 threshold)
# 2. KMeans embedding clustering
# 3. LLM semantic clustering (Joe ↔ Joseph)
```

### Assertion Challenging

```python
# Challenge a fact against existing knowledge
result = memory.challenge("Paris is the capital of Germany")

if result.has_conflicts:
    for conflict in result.conflicts:
        print(f"Contradicts: {conflict.existing_fact}")
        print(f"Method: {conflict.explanation}")  # [LLM], [Graph], [Embedding], etc.

# Auto-challenge during ingestion (smart triggering)
memory.ingest("The speed of light is 300,000 km/s")  # Challenges factual claims

# Control challenge behavior
memory.ingest(content, auto_challenge=True)   # Always challenge
memory.ingest(content, auto_challenge=False)  # Never challenge
```

### Temporal Versioning

```python
# Get version history for an item
versions = memory.version_tracker.get_versions(item_id)

# Get version at specific time (time-travel query)
from datetime import datetime
version = memory.version_tracker.get_version_at_time(
    item_id, 
    time=datetime(2024, 1, 15),
    time_type='transaction'
)

# Compare versions
diff = memory.version_tracker.compare_versions(item_id, version1=1, version2=2)
```

### Ontology Management

```python
from smartmemory.ontology import OntologyManager

# Load or create ontology
ontology_manager = OntologyManager()
ontology = ontology_manager.get_active_ontology()

# Ingest structured knowledge with pre-extracted entities
memory.ingest({
    "content": "John Smith works at Google",
    "entities": [
        {"name": "John Smith", "type": "Person"},
        {"name": "Google", "type": "Organization"}
    ],
    "relations": [
        {"source": "John Smith", "target": "Google", "type": "WORKS_AT"}
    ]
})
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
    
    def add_conversation(self, human_input, ai_response):
        # Store in both systems
        self.conversation_memory.save_context(
            {"input": human_input},
            {"output": ai_response}
        )
        
        # Add to SmartMemory for long-term retention
        self.smart_memory.ingest(f"User asked: {human_input}")
        self.smart_memory.ingest(f"I responded: {ai_response}")
```

### With OpenAI API

```python
import openai
from smartmemory import SmartMemory

class MemoryEnhancedChatbot:
    def __init__(self):
        self.memory = SmartMemory()
        
    def chat(self, user_message):
        # Search for relevant context
        context = self.memory.search(user_message, top_k=3)
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
        
        # Store conversation in memory (full pipeline)
        self.memory.ingest(f"User said: {user_message}")
        self.memory.ingest(f"I responded: {ai_response}")
        
        return ai_response
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize SmartMemory behavior
- [Core Concepts](../concepts/overview.md) - Understand the architecture
- [API Reference](../api/smart-memory.md) - Detailed API documentation
- [Advanced Features](../guides/advanced-features.md) - Explore powerful capabilities
