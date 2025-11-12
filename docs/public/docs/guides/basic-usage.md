# Basic Usage

This guide covers the fundamental operations and patterns for using SmartMemory effectively in your applications.

## Getting Started

### Initialize SmartMemory

```python
from smartmemory import SmartMemory

# Basic initialization
memory = SmartMemory()

# With configuration file
memory = SmartMemory(config_path="config.json")

# With inline configuration
memory = SmartMemory(config={
    "graph": {"backend": "FalkorDBBackend"},
    "background_processing": {"enabled": True}
})
```

## Adding Memories

### Simple Text Addition

The most basic way to add information to SmartMemory:

```python
# Add simple facts
memory.add("Python is a programming language")
memory.add("The capital of France is Paris")
memory.add("Machine learning is a subset of AI")

# Add personal experiences
memory.add("I learned Python programming in 2020")
memory.add("Had lunch with Sarah at the Italian restaurant yesterday")

# Add procedures
memory.add("To make coffee: heat water, add grounds, brew for 4 minutes")

# Add Zettelkasten notes (atomic knowledge)
memory.add(
    "Attention mechanisms allow models to focus on relevant input parts",
    memory_type="zettel",
    metadata={"tags": ["#ml", "#attention", "#transformer"]}
)
```

### Structured Data Addition

For more control over how memories are processed:

```python
# Add with explicit memory type
memory.add({
    "content": "Meeting with John about project timeline",
    "memory_type": "episodic",
    "metadata": {
        "participants": ["John"],
        "topic": "timeline",
        "urgency": "high"
    }
})

# Add with entities and relations
memory.add({
    "content": "John Smith works at Google",
    "entities": [
        {"name": "John Smith", "type": "PERSON"},
        {"name": "Google", "type": "ORGANIZATION"}
    ],
    "relations": [
        {"source": "John Smith", "target": "Google", "type": "WORKS_AT"}
    ]
})
```

### Batch Addition

For adding multiple memories efficiently:

```python
memories = [
    "Python is used for web development",
    "Django is a Python web framework",
    "Flask is another Python web framework",
    "FastAPI is a modern Python web framework"
]

# Add all memories
for memory_text in memories:
    memory.add(memory_text)

# Or use fast ingestion for background processing
for memory_text in memories:
    memory.ingest(memory_text)
```

### Adding Memories with Source Attribution (Grounding)

Grounding allows you to link memories to their sources for transparency and verification:

```python
# Add memory with source information
memory_id = memory.add("The Earth's circumference is approximately 40,075 km")

# Ground the memory to its source
memory.ground(
    item_id=memory_id,
    source_url="https://en.wikipedia.org/wiki/Earth",
    validation={"confidence": 0.95, "verified": True}
)

# Or add with source information directly
memory.add(
    content="Python was created by Guido van Rossum",
    context={
        "source_url": "https://python.org/about",
        "source_type": "official",
        "confidence": 0.98
    }
)
```

**Why use grounding?**
- **Transparency**: Track where information comes from
- **Fact-checking**: Verify claims against authoritative sources
- **Trust**: Build confidence in AI-generated responses
- **Audit trails**: Maintain records for compliance

## Searching Memories

### Basic Search

```python
# Search for relevant memories
results = memory.search("Python programming")

# Display results
for result in results:
    print(f"ID: {result.item_id}")
    print(f"Content: {result.content}")
    print(f"Type: {result.memory_type}")
    print("---")
```

### Type-Specific Search

```python
# Search only semantic memories (facts)
facts = memory.search("artificial intelligence", memory_type="semantic")

# Search only episodic memories (experiences)
experiences = memory.search("meeting", memory_type="episodic")

# Search only procedural memories (how-to)
procedures = memory.search("deploy", memory_type="procedural")
```

### Advanced Search Options

```python
# Limit number of results
top_results = memory.search("machine learning", top_k=3)

# User-specific search
user_memories = memory.search("project", user_id="user123")

# Search with context
results = memory.search(
    query="programming",
    memory_type="semantic",
    top_k=5
)
```

## Retrieving Specific Memories

### Get by ID

```python
# Get a specific memory
memory_item = memory.get("memory_id_123")

if memory_item:
    print(f"Content: {memory_item.content}")
    print(f"Created: {memory_item.created_at}")
    print(f"Type: {memory_item.memory_type}")
else:
    print("Memory not found")
```

### Get Related Memories

```python
# Find memories related to a specific memory
if memory_item:
    related = memory.get_neighbors(memory_item.item_id)
    print(f"Found {len(related)} related memories")
    
    for related_memory in related:
        print(f"- {related_memory.content}")
```

## Working with Relationships

### Automatic Relationships

SmartMemory automatically creates relationships based on:
- Semantic similarity
- Shared entities
- Temporal proximity
- Content overlap

```python
# Add related memories - SmartMemory will link them automatically
memory.add("Python is a programming language")
memory.add("I use Python for data analysis")
memory.add("Python has excellent machine learning libraries")

# Search for Python-related memories
python_memories = memory.search("Python")
for mem in python_memories:
    # Get automatically discovered relationships
    related = memory.get_neighbors(mem.item_id)
    print(f"'{mem.content}' has {len(related)} related memories")

# Create explicit relationships between memories
# Add memories first (add() returns item_id string)
python_memory_id = memory.add("Python programming language")
project_memory_id = memory.add("Data science project")

# Create explicit relationship
memory.link(
    source_id=python_memory_id,
    target_id=project_memory_id,
    link_type="USED_FOR"
)

# Get all links for a memory
links = memory.get_links(python_memory_id)
for link in links:
    print(link)  # Prints link string representation
```

### Semantic Memory (Facts & Knowledge)

Best for storing factual, timeless information:

```python
# Scientific facts
memory.add("Water boils at 100°C at sea level")
memory.add("The speed of light is 299,792,458 m/s")

# Technical knowledge
memory.add("REST APIs use HTTP methods like GET, POST, PUT, DELETE")
memory.add("Git is a distributed version control system")

# Search semantic knowledge
facts = memory.search("HTTP", memory_type="semantic")
```

### Episodic Memory (Experiences)

Best for personal experiences and events:

```python
# Personal events
memory.add("Attended the Python conference in San Francisco last month")
memory.add("Had a productive meeting with the development team yesterday")

# Learning experiences
memory.add("Completed the machine learning course on Coursera in March")
memory.add("Fixed the authentication bug after 3 hours of debugging")

# Search experiences
experiences = memory.search("conference", memory_type="episodic")
```

### Procedural Memory (How-to)

Best for step-by-step procedures and skills:

```python
# Technical procedures
memory.add("To deploy to AWS: 1) Build Docker image 2) Push to ECR 3) Update ECS service")
memory.add("Git workflow: create branch → make changes → commit → push → create PR")

# Daily procedures
memory.add("Morning routine: exercise → shower → coffee → check emails")

# Search procedures
procedures = memory.search("deploy", memory_type="procedural")
```

### Working Memory (Temporary)

Best for current context and temporary information:

```python
# Current tasks
memory.add("Currently debugging the authentication module")
memory.add("Working on user registration feature")

# Temporary variables
memory.add("Session token: abc123, expires in 1 hour")

# Search current context
current_work = memory.search("debugging", memory_type="working")
```

## Updating and Deleting Memories

### Update Existing Memories

```python
# Get existing memory
memory_item = memory.get("memory_id_123")

if memory_item:
    # Update content
    memory_item.content = "Updated content with new information"
    
    # Update metadata
    memory_item.metadata["status"] = "completed"
    
    # Save changes
    memory.update(memory_item)
    print(f"Updated item id: {memory_item.item_id}")

### Delete Memories

```python
# Delete specific memory
success = memory.delete("memory_id_123")
if success:
    print("Memory deleted successfully")
else:
    print("Failed to delete memory")

# Clear all memories (use with caution!)
memory.clear()
```

## Background Processing

### Background Processing

```python
# Synchronous (process now via full pipeline)
memory.ingest("Process immediately", sync=True)

# Asynchronous (quick persist + enqueue for background workers)
result = memory.ingest("Process in background", sync=False)
print(result)  # {"item_id": "...", "queued": True}

# Note: a separate worker service must consume the background queue.
```

### Monitor Background Processing

Background worker orchestration and monitoring are external to the core library. Use your queue system (e.g., Redis Streams) to implement metrics as needed.

## Error Handling

### Basic Error Handling

```python
from smartmemory.exceptions import MemoryError, BackendError

try:
    # Add memory
    result = memory.add("Some content")
    print(f"Added: {result.item_id}")
    
except MemoryError as e:
    print(f"Memory operation failed: {e}")
    
except BackendError as e:
    print(f"Backend error: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Search Error Handling

```python
try:
    results = memory.search("query")
    if not results:
        print("No memories found")
    else:
        print(f"Found {len(results)} memories")
        
except BackendError as e:
    print(f"Search failed: {e}")
    # Fallback to cached results or alternative search
```

## Performance Best Practices

### Efficient Memory Addition

```python
# Use background processing for non-critical additions
memory.ingest("Non-critical information")

# Use regular add() for critical information that needs immediate processing
critical_memory = memory.add("Critical information")

# Batch related additions
related_memories = [
    "Python programming",
    "Python web frameworks",
    "Python data science"
]

for mem in related_memories:
    memory.ingest(mem)  # Processed together in background
```

### Efficient Searching

```python
# Use specific memory types for better performance
semantic_results = memory.search("AI", memory_type="semantic")

# Limit results for faster response
quick_results = memory.search("programming", top_k=3)

# Cache frequently accessed memories
frequently_used = ["important_id_1", "important_id_2"]
cached_memories = {id: memory.get(id) for id in frequently_used}
```

### Memory Management

```python
# Get memory summary
stats = memory.summary()
print(f"Total memories: {stats.get('total_count', 0)}")

# Periodic cleanup (if needed)
# memory.clear()  # Use with extreme caution
```

## Integration Patterns

### With Web Applications

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
memory = SmartMemory()

@app.route('/add_memory', methods=['POST'])
def add_memory():
    content = request.json.get('content')
    item_id = memory.add(content)
    return jsonify({'id': item_id, 'status': 'added'})

@app.route('/search', methods=['GET'])
def search_memories():
    query = request.args.get('q')
    results = memory.search(query, top_k=10)
    return jsonify([{
        'id': r.item_id,
        'content': r.content,
        'type': r.memory_type
    } for r in results])
```

### With Data Processing Pipelines

```python
import smartmemory.utils


def process_documents(documents):
    """Process a batch of documents into memory"""
    for doc in documents:
        # Extract key information
        summary = extract_summary(doc)
        entities = extract_entities(doc)

        # Add to memory with structure
        memory.add({
            "content": summary,
            "memory_type": "semantic",
            "metadata": {
                "source": doc.source,
                "entities": entities,
                "processed_at": smartmemory.utils.now()
            }
        })
```

## Common Patterns

### Question-Answering System

```python
def answer_question(question):
    # Search for relevant memories
    relevant_memories = memory.search(question, top_k=5)
    
    if not relevant_memories:
        return "I don't have information about that."
    
    # Combine relevant information
    context = "\n".join([mem.content for mem in relevant_memories])
    
    # Generate answer (using LLM or rule-based approach)
    answer = generate_answer(question, context)
    
    # Store the Q&A as new memory
    memory.add(f"Question: {question}\nAnswer: {answer}")
    
    return answer
```

### Learning System

```python
class LearningSystem:
    def __init__(self):
        self.memory = SmartMemory()
    
    def learn_fact(self, fact):
        """Learn a new fact"""
        return self.memory.add(fact, memory_type="semantic")
    
    def record_experience(self, experience):
        """Record a personal experience"""
        return self.memory.add(experience, memory_type="episodic")
    
    def learn_procedure(self, procedure):
        """Learn a new procedure"""
        return self.memory.add(procedure, memory_type="procedural")
    
    def recall(self, query):
        """Recall relevant information"""
        return self.memory.search(query, top_k=5)
```

## Next Steps

- [Advanced Features](advanced-features) - Explore powerful capabilities
- [Ontology Management](ontology-management) - Structure your knowledge
- [Performance Tuning](performance-tuning) - Optimize for your use case
- [MCP Integration](mcp-integration) - Connect with LLM agents
