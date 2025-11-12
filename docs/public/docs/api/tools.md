# MCP Tools API Reference

SmartMemory provides a comprehensive set of Model Context Protocol (MCP) tools that enable seamless integration with agentic frameworks. This document provides detailed API documentation for all available MCP tools.

## Overview

The MCP tools interface provides standardized access to SmartMemory operations through the Model Context Protocol. These tools are automatically discoverable and can be used with any MCP-compatible framework.

## Available Tools

### Core Memory Operations

#### `mcp_memory_add`
Add new memory item to SmartMemory.

**Function Signature:**
```python
def mcp_memory_add(
    content: str,
    memory_type: str = "semantic",
    metadata: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters:**
- `content` (str): The content to store in memory
- `memory_type` (str, optional): Type of memory ("semantic", "episodic", "procedural", "working"). Default: "semantic"
- `metadata` (dict, optional): Additional metadata for the memory item

**Returns:**
- `str`: Unique identifier for the created memory item

**Example Usage:**
```python
from smartmemory.toolbox.finders.mcp_tools import mcp_memory_add

# Add semantic memory
memory_id = mcp_memory_add(
    content="Python is a versatile programming language",
    memory_type="semantic",
    metadata={
        "domain": "programming",
        "language": "python",
        "user_id": "user123"
    }
)
print(f"Created memory: {memory_id}")
```

**Tool Definition:**
```json
{
    "name": "mcp_memory_add",
    "description": "Add new memory item to SmartMemory with specified type and metadata",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to store in memory"
            },
            "memory_type": {
                "type": "string",
                "enum": ["semantic", "episodic", "procedural", "working"],
                "default": "semantic",
                "description": "Type of memory to create"
            },
            "metadata": {
                "type": "object",
                "description": "Additional metadata for the memory item"
            }
        },
        "required": ["content"]
    }
}
```

#### `mcp_memory_get`
Retrieve memory item by ID.

**Function Signature:**
```python
def mcp_memory_get(item_id: str) -> Optional[Dict[str, Any]]
```

**Parameters:**
- `item_id` (str): Unique identifier of the memory item

**Returns:**
- `dict` or `None`: Memory item data or None if not found

**Example Usage:**
```python
from smartmemory.toolbox.finders.mcp_tools import mcp_memory_get

# Retrieve memory by ID
memory_item = mcp_memory_get(item_id="memory_123")
if memory_item:
    print(f"Content: {memory_item['content']}")
    print(f"Type: {memory_item['memory_type']}")
    print(f"Created: {memory_item['created_at']}")
```

#### `mcp_memory_search`
Search memories using query and filters.

**Function Signature:**
```python
def mcp_memory_search(
    query: str,
    top_k: int = 5,
    memory_type: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]
```

**Parameters:**
- `query` (str): Search query string
- `top_k` (int, optional): Maximum number of results to return. Default: 5
- `memory_type` (str, optional): Filter by memory type
- `filters` (dict, optional): Additional search filters

**Returns:**
- `list`: List of matching memory items with relevance scores

**Example Usage:**
```python
from smartmemory.toolbox.finders.mcp_tools import mcp_memory_search

# Search for programming-related memories
results = mcp_memory_search(
    query="python programming best practices",
    top_k=10,
    memory_type="semantic",
    filters={
        "domain": "programming",
        "user_id": "user123"
    }
)

for result in results:
    print(f"Score: {result['score']:.2f} - {result['content'][:100]}...")
```

#### `mcp_memory_update`
Update existing memory item.

**Function Signature:**
```python
def mcp_memory_update(
    item_id: str,
    updates: Dict[str, Any]
) -> bool
```

**Parameters:**
- `item_id` (str): Unique identifier of the memory item
- `updates` (dict): Dictionary of updates to apply

**Returns:**
- `bool`: Success status

**Example Usage:**
```python
from smartmemory.toolbox.finders.mcp_tools import mcp_memory_update

# Update memory metadata
success = mcp_memory_update(
    item_id="memory_123",
    updates={
        "metadata.importance": 0.9,
        "metadata.tags": ["important", "reference"],
        "last_accessed": "2024-01-15T10:30:00Z"
    }
)
print(f"Update successful: {success}")
```

#### `mcp_memory_delete`
Delete memory item by ID.

**Function Signature:**
```python
def mcp_memory_delete(item_id: str) -> bool
```

**Parameters:**
- `item_id` (str): Unique identifier of the memory item

**Returns:**
- `bool`: Success status

**Example Usage:**
```python
from smartmemory.toolbox.finders.mcp_tools import mcp_memory_delete

# Delete memory item
success = mcp_memory_delete(item_id="memory_123")
print(f"Deletion successful: {success}")
```

#### `mcp_memory_ingest`
Full ingestion pipeline with enrichment and processing.

**Function Signature:**
```python
def mcp_memory_ingest(
    content: str,
    context: Optional[Dict[str, Any]] = None,
    options: Optional[Dict[str, Any]] = None
) -> str
```

**Parameters:**
- `content` (str): Content to ingest
- `context` (dict, optional): Contextual information for processing
- `options` (dict, optional): Processing options and configurations

**Returns:**
- `str`: Unique identifier for the ingested memory item

**Example Usage:**
```python
from smartmemory.toolbox.finders.mcp_tools import mcp_memory_ingest

# Full ingestion with context
memory_id = mcp_memory_ingest(
    content="Machine learning models require careful hyperparameter tuning",
    context={
        "user_id": "user123",
        "source": "conversation",
        "conversation_id": "conv_456"
    },
    options={
        "enable_entity_extraction": True,
        "enable_relationship_linking": True,
        "enable_enrichment": True,
        "background_processing": True
    }
)
print(f"Ingested memory: {memory_id}")
```

## Tool Discovery and Registration

### Automatic Discovery

MCP tools are automatically discovered through the toolbox system:

```python
from smartmemory.toolbox import get_tools

# Get all MCP memory tools
mcp_tools = get_tools(service="mcp")
print(f"Found {len(mcp_tools)} MCP tools:")

for tool in mcp_tools:
    print(f"- {tool.name}: {tool.description}")
```

### Manual Registration

Tools can also be manually registered:

```python
from smartmemory.toolbox.finders.mcp_tools import (
    mcp_memory_add,
    mcp_memory_search,
    mcp_memory_get,
    mcp_memory_update,
    mcp_memory_delete,
    mcp_memory_ingest
)

# Create tool registry
tool_registry = {
    "memory_add": mcp_memory_add,
    "memory_search": mcp_memory_search,
    "memory_get": mcp_memory_get,
    "memory_update": mcp_memory_update,
    "memory_delete": mcp_memory_delete,
    "memory_ingest": mcp_memory_ingest
}
```

## Framework Integration

### LangChain Integration

```python
from langchain.agents import Tool
from smartmemory.toolbox import get_tools

# Convert MCP tools to LangChain tools
def create_langchain_tools():
    mcp_tools = get_tools(service="mcp")
    langchain_tools = []
    
    for mcp_tool in mcp_tools:
        langchain_tool = Tool(
            name=mcp_tool.name,
            description=mcp_tool.description,
            func=mcp_tool.function,
            args_schema=mcp_tool.parameters
        )
        langchain_tools.append(langchain_tool)
    
    return langchain_tools

# Use with LangChain agent
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
tools = create_langchain_tools()

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Agent can now use memory tools
response = agent.run(
    "Remember that I prefer Python for data science projects and "
    "search for any existing preferences about programming languages"
)
```

### AutoGen Integration

```python
import autogen
from smartmemory.toolbox import get_tools

# Configure AutoGen with memory tools
def setup_autogen_with_memory():
    mcp_tools = get_tools(service="mcp")
    
    # Create tool definitions for AutoGen
    tool_definitions = []
    tool_functions = {}
    
    for tool in mcp_tools:
        tool_def = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
        }
        tool_definitions.append(tool_def)
        tool_functions[tool.name] = tool.function
    
    # Configure assistant with memory tools
    assistant = autogen.AssistantAgent(
        name="memory_assistant",
        llm_config={
            "models": "gpt-4",
            "tools": tool_definitions
        }
    )
    
    # Register tool functions
    for name, func in tool_functions.items():
        assistant.register_function(function_map={name: func})
    
    return assistant

assistant = setup_autogen_with_memory()
```

### Custom Framework Integration

```python
class CustomMemoryAgent:
    def __init__(self):
        # Get MCP tools
        self.memory_tools = {
            tool.name: tool.function 
            for tool in get_tools(service="mcp")
        }
    
    def process_with_memory(self, user_input, user_id):
        # Store user input
        input_memory_id = self.memory_tools["mcp_memory_add"](
            content=f"User input: {user_input}",
            memory_type="episodic",
            metadata={"user_id": user_id, "type": "input"}
        )
        
        # Search for relevant context
        context_memories = self.memory_tools["mcp_memory_search"](
            query=user_input,
            top_k=5,
            filters={"user_id": user_id}
        )
        
        # Process with context
        response = self.generate_response(user_input, context_memories)
        
        # Store response
        response_memory_id = self.memory_tools["mcp_memory_add"](
            content=f"Agent response: {response}",
            memory_type="episodic",
            metadata={"user_id": user_id, "type": "response"}
        )
        
        return response
```

## Advanced Tool Usage

### Batch Operations

```python
def batch_memory_operations(operations):
    """Execute multiple memory operations efficiently."""
    results = []
    
    for operation in operations:
        try:
            if operation["type"] == "add":
                result = mcp_memory_add(**operation["params"])
            elif operation["type"] == "search":
                result = mcp_memory_search(**operation["params"])
            elif operation["type"] == "update":
                result = mcp_memory_update(**operation["params"])
            elif operation["type"] == "delete":
                result = mcp_memory_delete(**operation["params"])
            elif operation["type"] == "ingest":
                result = mcp_memory_ingest(**operation["params"])
            else:
                result = {"error": f"Unknown operation: {operation['type']}"}
            
            results.append({
                "operation": operation["type"],
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "operation": operation["type"],
                "success": False,
                "error": str(e)
            })
    
    return results

# Example batch operations
operations = [
    {
        "type": "add",
        "params": {
            "content": "First memory",
            "memory_type": "semantic"
        }
    },
    {
        "type": "search",
        "params": {
            "query": "programming",
            "top_k": 3
        }
    }
]

results = batch_memory_operations(operations)
```

### Error Handling

```python
def safe_memory_operation(operation_name, **kwargs):
    """Safely execute memory operation with comprehensive error handling."""
    
    tool_map = {
        "add": mcp_memory_add,
        "get": mcp_memory_get,
        "search": mcp_memory_search,
        "update": mcp_memory_update,
        "delete": mcp_memory_delete,
        "ingest": mcp_memory_ingest
    }
    
    if operation_name not in tool_map:
        return {
            "success": False,
            "error": f"Unknown operation: {operation_name}",
            "error_type": "invalid_operation"
        }
    
    try:
        result = tool_map[operation_name](**kwargs)
        return {
            "success": True,
            "result": result,
            "operation": operation_name
        }
    except ValueError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "validation_error",
            "operation": operation_name
        }
    except ConnectionError as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "connection_error",
            "operation": operation_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": "unknown_error",
            "operation": operation_name
        }

# Usage with error handling
result = safe_memory_operation(
    "add",
    content="Test memory",
    memory_type="semantic"
)

if result["success"]:
    print(f"Memory created: {result['result']}")
else:
    print(f"Error ({result['error_type']}): {result['error']}")
```

## Tool Configuration

### Environment Configuration

```python
# Configure MCP tools through environment variables
import os

# SmartMemory configuration
os.environ["SMARTMEMORY_CONFIG"] = "/path/to/config.json"
os.environ["SMARTMEMORY_LOG_LEVEL"] = "INFO"

# Backend configuration (optional)
os.environ["FALKORDB_HOST"] = "localhost"
os.environ["FALKORDB_PORT"] = "6379"

# LLM provider configuration
os.environ["OPENAI_API_KEY"] = "your-api-key"
os.environ["ANTHROPIC_API_KEY"] = "your-api-key"
```

### Programmatic Configuration

```python
from smartmemory.toolbox.finders.mcp_tools import configure_mcp_tools

# Configure MCP tools with custom settings (aligned with SmartMemory configuration)
config = {
    "memory_config": {
        "graph_db": {"backend_class": "FalkorDBBackend"},
        "vector": {"backend": "chromadb", "persist_directory": ".chroma"},
        "background": {"enabled": True}
    },
    "tool_config": {
        "default_memory_type": "semantic",
        "max_search_results": 10,
        "enable_auto_metadata": True
    }
}

configure_mcp_tools(config)
```

## Performance Considerations

### Caching

```python
from functools import lru_cache

# Cache frequently accessed memories
@lru_cache(maxsize=100)
def cached_memory_get(item_id):
    """Get memory with caching for frequently accessed items."""
    return mcp_memory_get(item_id=item_id)

# Cache search results
@lru_cache(maxsize=50)
def cached_memory_search(query, memory_type=None, top_k=5):
    """Search with caching for common queries."""
    return mcp_memory_search(
        query=query,
        memory_type=memory_type,
        top_k=top_k
    )
```

### Async Operations

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def async_memory_operations(operations):
    """Execute memory operations asynchronously."""
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        loop = asyncio.get_event_loop()
        
        tasks = []
        for operation in operations:
            if operation["type"] == "add":
                task = loop.run_in_executor(
                    executor, 
                    mcp_memory_add, 
                    **operation["params"]
                )
            elif operation["type"] == "search":
                task = loop.run_in_executor(
                    executor, 
                    mcp_memory_search, 
                    **operation["params"]
                )
            # Add other operation types as needed
            
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

# Usage
operations = [
    {"type": "add", "params": {"content": "Memory 1"}},
    {"type": "add", "params": {"content": "Memory 2"}},
    {"type": "search", "params": {"query": "test"}}
]

results = asyncio.run(async_memory_operations(operations))
```

## Testing MCP Tools

### Unit Testing

```python
import unittest
from unittest.mock import patch, MagicMock

class TestMCPTools(unittest.TestCase):
    
    @patch('smartmemory.memory.smart_memory.SmartMemory')
    def test_mcp_memory_add(self, mock_memory):
        # Mock SmartMemory instance
        mock_instance = MagicMock()
        mock_memory.return_value = mock_instance
        mock_instance.add.return_value = "memory_123"
        
        # Test MCP add operation
        result = mcp_memory_add(
            content="Test content",
            memory_type="semantic"
        )
        
        self.assertEqual(result, "memory_123")
        mock_instance.add.assert_called_once()
    
    def test_tool_discovery(self):
        # Test that MCP tools are discoverable
        tools = get_tools(service="mcp")
        self.assertGreater(len(tools), 0)
        
        tool_names = [tool.name for tool in tools]
        expected_tools = [
            "mcp_memory_add",
            "mcp_memory_get",
            "mcp_memory_search",
            "mcp_memory_update",
            "mcp_memory_delete",
            "mcp_memory_ingest"
        ]
        
        for expected in expected_tools:
            self.assertIn(expected, tool_names)

if __name__ == "__main__":
    unittest.main()
```

## Best Practices

### Tool Usage Guidelines

1. **Use Appropriate Memory Types**: Choose the right memory type for your data
2. **Include Relevant Metadata**: Add metadata to improve searchability
3. **Handle Errors Gracefully**: Implement proper error handling
4. **Cache Frequent Operations**: Use caching for performance
5. **Batch When Possible**: Group related operations for efficiency

### Security Considerations

1. **Validate Inputs**: Always validate user inputs before processing
2. **Sanitize Content**: Clean content to prevent injection attacks
3. **Access Control**: Implement proper user access controls
4. **Audit Logging**: Log all memory operations for security auditing

### Performance Tips

1. **Limit Search Results**: Use reasonable `top_k` values
2. **Use Filters**: Apply filters to reduce search scope
3. **Async for Bulk**: Use async operations for bulk processing
4. **Monitor Usage**: Track tool usage patterns for optimization

## Next Steps

- **Integration Examples**: See MCP tools in action in [MCP Integration Guide](../guides/mcp-integration.md)
- **Advanced Features**: Explore [advanced features](../guides/advanced-features.md) for custom tool behaviors
- **Core API**: Complete SmartMemory API in [SmartMemory API](smart-memory.md)
- **Performance**: Optimize tool usage with [performance tuning](../guides/performance-tuning.md)

The MCP tools provide a standardized, framework-agnostic interface for integrating SmartMemory into any agentic system. They handle the complexity of memory operations while providing simple, consistent APIs for developers.
