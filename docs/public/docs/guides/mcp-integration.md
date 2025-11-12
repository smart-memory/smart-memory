# MCP Integration Guide

The Model Context Protocol (MCP) integration provides a standardized interface for agentic workflows to interact with SmartMemory. This guide covers how to use MCP tools and integrate SmartMemory with various AI frameworks.

## Overview

SmartMemory provides MCP tools that enable:
- **Standardized Memory Operations**: CRUD operations through MCP protocol
- **Agent Integration**: Easy integration with LLM agents and frameworks
- **Tool-based Interface**: Memory operations as callable tools
- **Service Discovery**: Auto-discovery of memory capabilities

## Available MCP Tools

### Core Memory Tools

SmartMemory exposes 7 primary MCP tools for memory operations:

| Tool | Purpose | Parameters |
|------|---------|------------|
| `mcp_memory_add` | Add new memory item | `content`, `memory_type`, `metadata` |
| `mcp_memory_get` | Retrieve memory by ID | `item_id` |
| `mcp_memory_search` | Search memories | `query`, `top_k`, `memory_type` |
| `mcp_memory_update` | Update existing memory | `item_id`, `updates` |
| `mcp_memory_delete` | Delete memory | `item_id` |
| `mcp_memory_ingest` | Full ingestion pipeline | `content`, `context`, `options` |
| `mcp_ground_memory` | Ground memory to source | `item_id`, `source_url`, `validation` |

### Tool Discovery

Tools are automatically discovered through the toolbox system:

```python
from smartmemory.toolbox import get_tools

# Get all MCP memory tools
mcp_tools = get_tools(service="mcp")
print(f"Found {len(mcp_tools)} MCP tools")

# Get specific tool
memory_add_tool = get_tools(name="mcp_memory_add")[0]
```

## Basic Usage

### Setting Up MCP Integration

```python
from smartmemory.stores.external.mcp_handler import MCPHandler
from smartmemory.smart_memory import SmartMemory

# Initialize SmartMemory
memory = SmartMemory()

# Create MCP handler
mcp_handler = MCPHandler(memory)

# Use MCP operations
result = mcp_handler.add({
    "content": "Important meeting notes",
    "memory_type": "episodic",
    "metadata": {"user_id": "user123", "priority": "high"}
})
```

### Using MCP Tools Directly

```python
from smartmemory.toolbox.finders.mcp_tools import (
    mcp_memory_add, mcp_memory_search, mcp_memory_get
)

# Add a memory
add_result = mcp_memory_add(
    content="Project deadline is next Friday",
    memory_type="episodic",
    metadata={"project": "alpha", "user_id": "user123"}
)
print(f"Added memory: {add_result}")

# Search memories
search_results = mcp_memory_search(
    query="project deadline",
    top_k=5,
    memory_type="episodic"
)
print(f"Found {len(search_results)} relevant memories")

# Get specific memory
if search_results:
    memory_id = search_results[0]["id"]
    memory_item = mcp_memory_get(item_id=memory_id)
    print(f"Retrieved: {memory_item}")
```

### Grounding with MCP Tools

Establish provenance for memories using the grounding MCP tool:

```python
from smartmemory.toolbox.finders.mcp_tools import (
    mcp_memory_add, mcp_ground_memory
)

# Add a factual memory
add_result = mcp_memory_add(
    content="The speed of light in vacuum is 299,792,458 meters per second",
    memory_type="semantic",
    metadata={"topic": "physics", "importance": "high"}
)

# Extract the memory ID from the result
memory_id = add_result["id"]  # Assuming the result contains the ID

# Ground the memory to its authoritative source
ground_result = mcp_ground_memory(
    item_id=memory_id,
    source_url="https://physics.nist.gov/cgi-bin/cuu/Value?c",
    validation={
        "confidence": 0.99,
        "source_authority": "NIST",
        "verified_at": "2024-01-15T10:30:00Z",
        "verification_method": "official_standard"
    }
)
print(f"Grounding result: {ground_result}")
```

#### Agent-Driven Grounding Workflow

AI agents can automatically ground memories during conversations:

```python
import smartmemory.utils


def agent_with_grounding_workflow(user_query, sources_context):
    """Example agent workflow that adds and grounds memories"""

    # Agent processes user query and identifies factual claims
    claims = extract_factual_claims(user_query)  # Custom extraction

    for claim in claims:
        # Add the claim as a memory
        memory_result = mcp_memory_add(
            content=claim["text"],
            memory_type="semantic",
            metadata={"claim_type": claim["type"], "user_context": True}
        )

        # If source is available, ground the memory
        if claim.get("source_url"):
            mcp_ground_memory(
                item_id=memory_result["id"],
                source_url=claim["source_url"],
                validation={
                    "confidence": claim.get("confidence", 0.8),
                    "extraction_method": "ai_agent",
                    "verified_at": smartmemory.utils.now().isoformat()
                }
            )

    return f"Added and grounded {len(claims)} factual claims"
```

## Agent Framework Integration

### LangChain Integration

```python
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from smartmemory.toolbox import get_tools

# Get MCP memory tools
memory_tools = get_tools(service="mcp")

# Convert to LangChain tools
langchain_tools = []
for tool in memory_tools:
    langchain_tools.append(Tool(
        name=tool.name,
        description=tool.description,
        func=tool.function
    ))

# Initialize agent with memory tools
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=langchain_tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# Use agent with memory capabilities
response = agent.run(
    "Remember that I prefer morning meetings and search for any "
    "existing scheduling preferences"
)
```

### AutoGen Integration

```python
import autogen
from smartmemory.toolbox import get_tools

# Get memory tools
memory_tools = get_tools(service="mcp")

# Create tool registry for AutoGen
tool_registry = {}
for tool in memory_tools:
    tool_registry[tool.name] = tool.function

# Configure assistant with memory tools
assistant = autogen.AssistantAgent(
    name="memory_assistant",
    llm_config={
        "models": "gpt-4",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            for tool in memory_tools
        ]
    }
)

# Register tool functions
for name, func in tool_registry.items():
    assistant.register_function(function_map={name: func})
```

### Custom Agent Integration

```python
class MemoryEnhancedAgent:
    def __init__(self):
        self.memory_tools = {
            tool.name: tool.function 
            for tool in get_tools(service="mcp")
        }
    
    def process_message(self, message, user_id):
        # Store conversation in memory
        self.memory_tools["mcp_memory_add"](
            content=f"User message: {message}",
            memory_type="episodic",
            metadata={"user_id": user_id, "type": "conversation"}
        )
        
        # Search for relevant context
        context = self.memory_tools["mcp_memory_search"](
            query=message,
            top_k=3,
            memory_type="semantic"
        )
        
        # Generate response with context
        response = self.generate_response(message, context)
        
        # Store response
        self.memory_tools["mcp_memory_add"](
            content=f"Agent response: {response}",
            memory_type="episodic",
            metadata={"user_id": user_id, "type": "response"}
        )
        
        return response
```

## Advanced Usage Patterns

### Contextual Memory Operations

```python
import smartmemory.utils


def contextual_memory_operation(user_id, operation, **kwargs):
    """Add user context to all memory operations."""

    # Add user context to metadata
    if "metadata" not in kwargs:
        kwargs["metadata"] = {}
    kwargs["metadata"]["user_id"] = user_id
    kwargs["metadata"]["timestamp"] = smartmemory.utils.now().isoformat()

    # Route to appropriate tool
    tool_map = {
        "add": mcp_memory_add,
        "search": mcp_memory_search,
        "get": mcp_memory_get,
        "update": mcp_memory_update,
        "delete": mcp_memory_delete,
        "ingest": mcp_memory_ingest
    }

    if operation in tool_map:
        return tool_map[operation](**kwargs)
    else:
        raise ValueError(f"Unknown operation: {operation}")


# Usage
result = contextual_memory_operation(
    user_id="user123",
    operation="add",
    content="Meeting with client tomorrow",
    memory_type="episodic"
)
```

### Batch Operations

```python
def batch_memory_operations(operations):
    """Execute multiple memory operations in batch."""
    results = []
    
    for op in operations:
        try:
            if op["type"] == "add":
                result = mcp_memory_add(**op["params"])
            elif op["type"] == "search":
                result = mcp_memory_search(**op["params"])
            elif op["type"] == "update":
                result = mcp_memory_update(**op["params"])
            else:
                result = {"error": f"Unknown operation: {op['type']}"}
            
            results.append({
                "operation": op["type"],
                "success": True,
                "result": result
            })
        except Exception as e:
            results.append({
                "operation": op["type"],
                "success": False,
                "error": str(e)
            })
    
    return results

# Usage
operations = [
    {
        "type": "add",
        "params": {
            "content": "First memory",
            "memory_type": "semantic"
        }
    },
    {
        "type": "add",
        "params": {
            "content": "Second memory",
            "memory_type": "episodic"
        }
    },
    {
        "type": "search",
        "params": {
            "query": "memory",
            "top_k": 5
        }
    }
]

results = batch_memory_operations(operations)
```

### Memory-Aware Conversation Flow

```python
import smartmemory.utils


class MemoryAwareConversation:
    def __init__(self, user_id):
        self.user_id = user_id
        self.conversation_id = str(uuid.uuid4())

    def add_message(self, role, content):
        """Add message to conversation memory."""
        return mcp_memory_add(
            content=content,
            memory_type="episodic",
            metadata={
                "user_id": self.user_id,
                "conversation_id": self.conversation_id,
                "role": role,
                "timestamp": smartmemory.utils.now().isoformat()
            }
        )

    def get_relevant_context(self, query, max_context=3):
        """Get relevant memories for current query."""
        return mcp_memory_search(
            query=query,
            top_k=max_context,
            memory_type="semantic"
        )

    def get_conversation_history(self, limit=10):
        """Get recent conversation history."""
        return mcp_memory_search(
            query=f"conversation_id:{self.conversation_id}",
            top_k=limit,
            memory_type="episodic"
        )

    def process_turn(self, user_message):
        """Process a conversation turn with memory."""
        # Store user message
        self.add_message("user", user_message)

        # Get relevant context
        context = self.get_relevant_context(user_message)

        # Get conversation history
        history = self.get_conversation_history()

        # Generate response (placeholder)
        response = self.generate_response(user_message, context, history)

        # Store response
        self.add_message("assistant", response)

        return response, context
```

## Error Handling and Best Practices

### Robust Error Handling

```python
def safe_memory_operation(operation, **kwargs):
    """Safely execute memory operation with error handling."""
    try:
        tool_map = {
            "add": mcp_memory_add,
            "search": mcp_memory_search,
            "get": mcp_memory_get,
            "update": mcp_memory_update,
            "delete": mcp_memory_delete,
            "ingest": mcp_memory_ingest
        }
        
        if operation not in tool_map:
            return {"success": False, "error": f"Unknown operation: {operation}"}
        
        result = tool_map[operation](**kwargs)
        return {"success": True, "result": result}
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "operation": operation,
            "params": kwargs
        }

# Usage
result = safe_memory_operation(
    "add",
    content="Test memory",
    memory_type="semantic"
)

if result["success"]:
    print(f"Operation successful: {result['result']}")
else:
    print(f"Operation failed: {result['error']}")
```

### Performance Optimization

```python
# Use batch operations for multiple items
def optimized_bulk_add(items):
    """Efficiently add multiple items."""
    results = []
    
    # Group by memory type for better performance
    by_type = {}
    for item in items:
        memory_type = item.get("memory_type", "semantic")
        if memory_type not in by_type:
            by_type[memory_type] = []
        by_type[memory_type].append(item)
    
    # Process each type in batch
    for memory_type, type_items in by_type.items():
        for item in type_items:
            result = mcp_memory_add(**item)
            results.append(result)
    
    return results

# Cache frequently accessed memories
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_memory_get(item_id):
    """Get memory with caching."""
    return mcp_memory_get(item_id=item_id)
```

## Testing MCP Integration

### Unit Testing

```python
import unittest
from unittest.mock import patch, MagicMock

class TestMCPIntegration(unittest.TestCase):
    
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
```

### Integration Testing

```python
def test_end_to_end_mcp_workflow():
    """Test complete MCP workflow."""
    
    # Add memory
    add_result = mcp_memory_add(
        content="Integration test memory",
        memory_type="semantic",
        metadata={"test": True}
    )
    
    assert add_result is not None
    memory_id = add_result
    
    # Search for memory
    search_results = mcp_memory_search(
        query="integration test",
        top_k=5
    )
    
    assert len(search_results) > 0
    assert any(item["id"] == memory_id for item in search_results)
    
    # Get specific memory
    retrieved = mcp_memory_get(item_id=memory_id)
    assert retrieved is not None
    assert "integration test" in retrieved["content"].lower()
    
    # Update memory
    update_result = mcp_memory_update(
        item_id=memory_id,
        updates={"metadata": {"test": True, "updated": True}}
    )
    assert update_result is not None
    
    # Delete memory
    delete_result = mcp_memory_delete(item_id=memory_id)
    assert delete_result is not None
    
    print("End-to-end MCP workflow test passed!")
```

## Troubleshooting

### Common Issues

1. **Tool Not Found**
   ```python
   # Check if tools are properly registered
   tools = get_tools(service="mcp")
   if not tools:
       print("No MCP tools found. Check installation.")
   ```

2. **Memory Operation Fails**
   ```python
   # Check SmartMemory initialization
   try:
       result = mcp_memory_add(content="test")
   except Exception as e:
       print(f"Memory operation failed: {e}")
   ```

3. **Missing Dependencies**
   ```python
   # Check for required packages
   try:
       from smartmemory.toolbox.finders.mcp_tools import mcp_memory_add
   except ImportError as e:
       print(f"Missing dependency: {e}")
   ```

### Debug Mode

```python
import logging

# Enable debug logging for MCP operations
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("smartmemory.mcp")

# Use debug wrapper
def debug_mcp_operation(operation, **kwargs):
    logger.debug(f"Executing MCP operation: {operation} with {kwargs}")
    result = operation(**kwargs)
    logger.debug(f"Operation result: {result}")
    return result

# Usage
result = debug_mcp_operation(
    mcp_memory_add,
    content="Debug test",
    memory_type="semantic"
)
```

## Next Steps

- **Explore Advanced Features**: Learn about [ontology management](ontology-management.md)
- **Performance Tuning**: Optimize for your use case with [performance tuning](performance-tuning.md)
- **Custom Integration**: Build custom agents with [advanced features](advanced-features.md)
- **API Reference**: Detailed API documentation in [MCP Tools API](../api/tools.md)

The MCP integration provides a powerful, standardized way to incorporate SmartMemory into any agentic workflow. Start with basic operations and gradually explore advanced patterns as your needs grow.
