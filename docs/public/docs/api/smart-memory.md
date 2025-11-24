The `SmartMemory` class is the main entry point for all memory operations. It provides a unified interface for storing, retrieving, and managing different types of memories with intelligent processing capabilities.

## Class: SmartMemory

  ```python
  from smartmemory import SmartMemory

  memory = SmartMemory(
      config_path: Optional[str] = None,
      config: Optional[Dict] = None,
      **kwargs
  )
  ```

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `Optional[str]` | `None` | Path to JSON configuration file |
| `config` | `Optional[Dict]` | `None` | Configuration dictionary (overrides file) |
| `**kwargs` | `Any` | - | Additional configuration options |

### Example

```python
# Basic initialization
memory = SmartMemory()

# With custom configuration
memory = SmartMemory(config_path="config.json")

# With inline configuration
memory = SmartMemory(config={
    "graph": {"backend": "FalkorDBBackend"}
})
```

## Core Methods

### ingest()

Ingest content with full intelligent processing pipeline (extract → store → link → enrich → evolve).

```python
def ingest(
    self,
    item: Union[str, Dict, MemoryItem],
    context: Optional[Dict] = None,
    adapter_name: Optional[str] = None,
    converter_name: Optional[str] = None,
    extractor_name: Optional[str] = None,
    enricher_names: Optional[List[str]] = None,
    conversation_context: Optional[Union[ConversationContext, Dict[str, Any]]] = None,
    user_id: Optional[str] = None,
    sync: Optional[bool] = None,
    **kwargs
) -> Union[str, Dict[str, Any]]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `item` | `Union[str, Dict, MemoryItem]` | Content to ingest |
| `context` | `Optional[Dict]` | Additional context for processing |
| `adapter_name` | `Optional[str]` | Specific adapter to use |
| `converter_name` | `Optional[str]` | Specific converter to use |
| `extractor_name` | `Optional[str]` | Specific extractor to use |
| `enricher_names` | `Optional[List[str]]` | Specific enrichers to use |
| `conversation_context` | `Optional[Union[ConversationContext, Dict]]` | Conversation context |
| `user_id` | `Optional[str]` | User ID for user isolation |
| `sync` | `Optional[bool]` | If True (default), run synchronously. If False, queue for background processing |
| `**kwargs` | `Any` | Additional processing options |

#### Returns

- `str` - The item_id (when sync=True)
- `Dict[str, Any]` - `{"item_id": str, "queued": bool}` (when sync=False)

#### Examples

```python
# Simple ingestion (full pipeline)
item_id = memory.ingest("I learned Python programming in 2020")

# With user isolation
item_id = memory.ingest(
    "Meeting with John about project timeline",
    user_id="alice"
)

# Async ingestion (queue for background)
result = memory.ingest("Large document...", sync=False)
print(f"Queued: {result['item_id']}, queued: {result['queued']}")

# With conversation context
from smartmemory.conversation.context import ConversationContext
conv_ctx = ConversationContext(conversation_id="conv_123")
item_id = memory.ingest(
    "User prefers Python for data science",
    conversation_context=conv_ctx
)
```

---

### add()

Simple storage without the full ingestion pipeline. Use for internal operations, derived items, or when pipeline is not needed.

```python
def add(
    self,
    item: Union[str, Dict, MemoryItem],
    **kwargs
) -> str  # item_id
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `item` | `Union[str, Dict, MemoryItem]` | Content to store |
| `**kwargs` | `Any` | Additional storage options |

#### Returns

`str` - The item_id of the stored memory item

#### Examples

```python
from smartmemory import MemoryItem

# Simple storage (no extraction/linking/evolution)
item = MemoryItem(content="Quick note", memory_type="semantic")
item_id = memory.add(item)

# With metadata
item = MemoryItem(
    content="Derived fact",
    memory_type="semantic",
    metadata={"source": "enrichment"}
)
item_id = memory.add(item)
```

**When to use which:**
- `ingest()` - User input, external data, anything needing full processing
- `add()` - Internal operations, derived items, evolution/enrichment output

### search()

Search for memories using various strategies.

```python
def search(
    self,
    query: str,
    top_k: int = 5,
    memory_type: Optional[str] = None
) -> List[MemoryItem]
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | - | Search query string |
| `top_k` | `int` | `5` | Maximum number of results |
| `memory_type` | `Optional[str]` | `None` | Filter by memory type |

**Note:** User/tenant filtering is handled automatically by `ScopeProvider`. No manual `user_id` parameter needed.

#### Returns

`List[MemoryItem]` - List of matching memory items

#### Examples

```python
# Basic search
results = memory.search("Python programming")

# Type-specific search
semantic_results = memory.search("France", memory_type="semantic")
episodic_results = memory.search("meeting", memory_type="episodic")

# Limited results
top_results = memory.search("AI", top_k=3)
```

### get()

Retrieve a specific memory item by ID.

```python
def get(self, key: str) -> Optional[MemoryItem]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Memory item ID |

#### Returns

`Optional[MemoryItem]` - The memory item if found, None otherwise

#### Example

```python
# Get specific memory
memory_item = memory.get("memory_id_123")
if memory_item:
    print(f"Content: {memory_item.content}")
```

### update()

Update an existing memory item.

```python
def update(self, item: Union[MemoryItem, Dict]) -> str  # item_id
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `item` | `Union[MemoryItem, Dict]` | Updated memory item or data |

#### Returns

`str` - The item_id of the updated item

#### Example

```python
# Update existing memory
memory_item = memory.get("memory_id_123")
memory_item.content = "Updated content"
updated_item = memory.update(memory_item)
```

### delete()

Delete a memory item.

```python
def delete(self, key: str) -> bool
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `key` | `str` | Memory item ID to delete |

#### Returns

`bool` - True if deleted successfully, False otherwise

#### Example

```python
# Delete memory
success = memory.delete("memory_id_123")
print(f"Deleted: {success}")
```

## Relationship Methods

### link()

Create explicit relationships between memories.

```python
def link(
    self,
    source_id: str,
    target_id: str,
    link_type: Union[str, LinkType] = "RELATED"
) -> str  # "Linked <src> to <tgt> as <type>"
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source_id` | `str` | - | Source memory ID |
| `target_id` | `str` | - | Target memory ID |
| `link_type` | `Union[str, LinkType]` | `"RELATED"` | Type of relationship |

#### Returns

`str` - Confirmation string

#### Example

```python
# Create relationship
memory.link("python_memory", "project_memory", "USED_FOR")
```

### get_links()

Get all links for a memory item.

```python
def get_links(
    self,
    item_id: str,
    memory_type: str = "semantic"
) -> List[str]  # List of link/triple strings
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `item_id` | `str` | - | Memory item ID |
| `memory_type` | `str` | `"semantic"` | Memory type context |

#### Returns

`List[str]` - List of link/triple strings for all edges involving item_id

#### Example

```python
# Get all links
links = memory.get_links("memory_id_123")
for link in links:
    print(link)  # Prints link string representation
```

### get_neighbors()

Get neighboring memories through relationships.

```python
def get_neighbors(self, item_id: str) -> List[MemoryItem]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `item_id` | `str` | Memory item ID |

#### Returns

`List[MemoryItem]` - List of neighboring memory items

#### Example

```python
# Get related memories
neighbors = memory.get_neighbors("memory_id_123")
print(f"Found {len(neighbors)} related memories")
```

## Background Processing

SmartMemory supports synchronous and asynchronous ingestion via the `ingest()` method.

```python
# Synchronous (process now via full pipeline)
memory.ingest("Process immediately", sync=True)

# Asynchronous (quick persist + enqueue for background workers)
result = memory.ingest("Process in background", sync=False)
print(result)  # {"item_id": "...", "queued": True}

# Note: a separate worker service must consume the background queue.
```

## Utility Methods

### ground()

Ground a memory item to an external source for provenance and validation.

```python
def ground(
    self,
    item_id: str,
    source_url: str,
    validation: Optional[Dict] = None
) -> None
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `item_id` | `str` | ID of the memory item to ground |
| `source_url` | `str` | URL of the external source for provenance |
| `validation` | `Optional[Dict]` | Optional validation metadata |

#### Description

Grounding establishes provenance by linking a memory item to its external source. This is crucial for:
- **Fact verification**: Link memories to authoritative sources
- **Audit trails**: Track the origin of information
- **Quality assurance**: Associate validation metadata
- **Transparency**: Enable source traceability for AI decisions

#### Example

```python
# Ground a memory to its source
memory.ground(
    item_id="fact_123",
    source_url="https://example.com/article",
    validation={"confidence": 0.95, "verified_at": "2024-01-01"}
)

# Ground with minimal information
memory.ground("memory_456", "https://source.com/data")
```

### clear()

Clear all memory data.

```python
def clear(self) -> None
```

#### Example

```python
# Clear all memories (use with caution!)
memory.clear()
```

### resolve_external()

Resolve external references in a memory item.

```python
def resolve_external(self, node: MemoryItem) -> Optional[List[Any]]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `node` | `MemoryItem` | Memory item with external references |

#### Returns

`Optional[List[Any]]` - Resolution results or None

## Ingest

```python
def ingest(
    self,
    item: Any,
    context: Optional[Dict] = None,
    adapter_name: Optional[str] = None,
    converter_name: Optional[str] = None,
    extractor_name: Optional[str] = None,
    sync: Optional[bool] = None,
    conversation_context: Optional[Union[ConversationContext, Dict[str, Any]]] = None,
    user_id: Optional[str] = None
) -> Union[Dict[str, Any], Any]
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `item` | `Any` | Content to ingest |
| `context` | `Optional[Dict]` | Additional context for processing |
| `adapter_name` | `Optional[str]` | Specific adapter to use |
| `converter_name` | `Optional[str]` | Specific converter to use |
| `extractor_name` | `Optional[str]` | Specific extractor to use |
| `sync` | `Optional[bool]` | If True, run full pipeline synchronously; if False, enqueue for background |
| `conversation_context` | `Optional[Union[ConversationContext, Dict]]` | Conversation context |
| `user_id` | `Optional[str]` | User ID for user isolation (sets item.user_id) |

#### Behavior

When `sync=True` (or configured for local mode), runs the full ingestion pipeline and returns the pipeline result.
When `sync=False`, quickly persists and enqueues background work, returning `{ "item_id": str, "queued": bool }`.

#### Examples

```python
# Synchronous ingestion with user isolation
result = memory.ingest(
    "Process this immediately",
    user_id="alice",
    sync=True
)

# Asynchronous ingestion
result = memory.ingest(
    "Process in background",
    user_id="alice",
    sync=False
)
print(result)  # {"item_id": "...", "queued": True}
```

## Error Handling

SmartMemory methods can raise the following exceptions:

### MemoryError
Raised when memory operations fail.

```python
from smartmemory.exceptions import MemoryError

try:
    result = memory.add("content")
except MemoryError as e:
    print(f"Memory operation failed: {e}")
```

### ConfigurationError
Raised when configuration is invalid.

```python
from smartmemory.exceptions import ConfigurationError

try:
    memory = SmartMemory(config_path="invalid.json")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### BackendError
Raised when backend operations fail.

```python
from smartmemory.exceptions import BackendError

try:
    results = memory.search("query")
except BackendError as e:
    print(f"Backend error: {e}")
```

## Configuration Properties

### Access Configuration

```python
# Configuration is set during initialization
memory = SmartMemory(config_path="config.json")

# Or with inline configuration
memory = SmartMemory(config={
    "graph": {"backend": "FalkorDBBackend"},
    "similarity": {"semantic_weight": 0.5}
})
```

**Note**: Configuration is set at initialization time. Runtime configuration updates are not supported.

## Performance Tips

1. **Batch operations** when possible:
   ```python
   items = ["item1", "item2", "item3"]
   for item in items:
       memory.ingest(item, sync=False)  # Enqueue for background workers
   ```

3. **Cache frequently accessed memories**:
   ```python
   # Store frequently used memory IDs
   important_memories = ["id1", "id2", "id3"]
   cached_memories = {id: memory.get(id) for id in important_memories}
   ```

4. **Use specific memory types** for better search performance:
   ```python
   # More efficient than general search
   results = memory.search("query", memory_type="semantic")
   ```

## Additional Methods

### Evolution and Memory Management

#### `run_evolution_cycle()`

Trigger a memory evolution cycle for consolidation and optimization.

```python
def run_evolution_cycle(self) -> None
```

**Example:**
```python
# Trigger evolution cycle
memory.run_evolution_cycle()
```

#### `commit_working_to_episodic()`

Commit working memory items to episodic memory.

```python
def commit_working_to_episodic(self, remove_from_source: bool = True) -> List[str]
```

**Returns:** List of committed item IDs

#### `commit_working_to_procedural()`

Commit working memory items to procedural memory.

```python
def commit_working_to_procedural(self, remove_from_source: bool = True) -> List[str]
```

**Returns:** List of committed item IDs

### Graph Operations

#### `add_edge()`

Add an edge between two memory nodes.

```python
def add_edge(
    self,
    source_id: str,
    target_id: str,
    relation_type: str,
    properties: Optional[Dict] = None
) -> Any
```

#### `create_or_merge_node()`

Create or merge a node with properties.

```python
def create_or_merge_node(
    self,
    item_id: str,
    properties: Dict,
    memory_type: Optional[str] = None
) -> str
```

### Monitoring and Analytics

#### `summary()`

Get summary statistics about memory contents.

```python
def summary(self) -> Dict
```

**Example:**
```python
stats = memory.summary()
print(f"Total items: {stats.get('total_count', 0)}")
```

#### `orphaned_notes()`

Find orphaned memory items (no connections).

```python
def orphaned_notes(self) -> List
```

#### `prune()`

Prune old or unused memories.

```python
def prune(self, strategy: str = "old", days: int = 365, **kwargs) -> Any
```

### Temporal Queries

#### `time_travel()`

Context manager for temporal queries at a specific point in time.

```python
def time_travel(self, to: str) -> ContextManager
```

**Example:**
```python
# Query memories as they existed on a specific date
with memory.time_travel("2024-09-01"):
    results = memory.search("Python")
    # Results reflect state on Sept 1st, 2024
```

### Archive Operations

#### `archive_put()`

Archive a conversation artifact durably.

```python
def archive_put(
    self,
    conversation_id: str,
    payload: Union[bytes, Dict[str, Any]],
    metadata: Dict[str, Any]
) -> Dict[str, str]
```

**Returns:** Dictionary with `archive_uri` and `content_hash`

#### `archive_get()`

Retrieve an archived artifact by URI.

```python
def archive_get(self, archive_uri: str) -> Union[bytes, Dict[str, Any]]
```

### Tags and Metadata

#### `add_tags()`

Add tags to a memory item.

```python
def add_tags(self, item_id: str, tags: List[str]) -> bool
```

**Example:**
```python
memory.add_tags("memory_123", ["important", "reference", "python"])
```

#### `update_properties()`

Update memory node properties with merge/replace semantics.

```python
def update_properties(
    self,
    item_id: str,
    properties: Dict,
    write_mode: Optional[str] = None
) -> Any
```

## Thread Safety

SmartMemory is thread-safe for read operations but requires coordination for write operations:

```python
import threading

# Safe for concurrent reads
def read_worker():
    results = memory.search("query")
    return results

# Coordinate writes
write_lock = threading.Lock()

def write_worker(content):
    with write_lock:
        memory.add(content)
```

## Next Steps

- [Memory Types API](memory-types.md) - Specific memory type interfaces
- [Components API](components.md) - Component-level APIs
- [Tools API](tools.md) - MCP tools and integrations
- [Factories API](factories.md) - Factory pattern APIs
