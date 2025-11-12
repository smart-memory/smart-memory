# Configuration

SmartMemory provides extensive configuration options to customize behavior for your specific use case. This guide covers all available configuration options and best practices.

## Configuration File Structure

SmartMemory uses JSON configuration files with environment variable expansion support:

```json
{
  "graph_db": {
    "backend_class": "FalkorDBBackend",
    "host": "localhost",
    "port": 6379,
    "graph_name": "smartmemory"
  },
  "vector": {
    "backend": "falkordb",
    "persist_directory": ".chroma"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}"
  },
  "extractor": {
    "spacy_model": "en_core_web_sm"
  },
  "background": {
    "enabled": true,
    "max_workers": 3,
    "queue_size": 1000
  },
  "similarity": {
    "semantic_weight": 0.4,
    "content_weight": 0.3,
    "temporal_weight": 0.2,
    "metadata_weight": 0.1
  },
  "evolution": {
    "enabled": true,
    "algorithms": [
      "MaximalConnectivityEvolver",
      "RapidEnrichmentEvolver",
      "StrategicPruningEvolver"
    ]
  }
}
```

## Graph Backend Configuration

### FalkorDB Backend (Recommended)

```json
{
  "graph_db": {
    "backend_class": "FalkorDBBackend",
    "host": "localhost",
    "port": 6379,
    "graph_name": "smartmemory",
    "password": "${REDIS_PASSWORD}",
    "ssl": false,
    "ssl_cert_reqs": "none"
  }
}
```

### Neo4j Backend

```json
{
  "graph_db": {
    "backend_class": "Neo4jBackend",
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "${NEO4J_PASSWORD}",
    "database": "smartmemory"
  }
}
```

<!-- In-memory backend is not supported in the current release; use FalkorDB or Neo4j. -->

## Vector Store Configuration

SmartMemory defaults to FalkorDB for vector storage and search.

### FalkorDB (Default)

```json
{
  "vector": {
    "default": "falkordb",
    "type": "falkordb",
    "dimension": 1536,
    "metric": "cosine",
    "hnsw_m": 16,
    "hnsw_ef_construction": 200,
    "hnsw_ef_runtime": 64
  }
}
```

Backend writes embeddings as native vectors and creates a Vector Index automatically. Minimal Cypher (for reference):

```cypher
// Schema
CREATE VECTOR INDEX FOR (n:Vec_default) ON (n.embedding)
OPTIONS {dimension:1536, similarityFunction:'cosine', M:16, efConstruction:200, efRuntime:64};

// Insert
CREATE (:Vec_default {id:'a', embedding: vecf32([1.0, 0.0])});

// Search top-8
CALL db.idx.vector.queryNodes('Vec_default','embedding', 8, vecf32([0.9, 0.1]))
YIELD node, score
RETURN node.id, score
ORDER BY score DESC;
```

Knobs: `vector.dimension`, `vector.metric` ('cosine'|'euclidean'), `vector.hnsw_m`, `vector.hnsw_ef_construction`, `vector.hnsw_ef_runtime`.

### ChromaDB (Optional)

```json
{
  "vector": {
    "backend": "chromadb",
    "persist_directory": ".chroma"
  }
}
```

## LLM Configuration

### OpenAI

```json
{
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}",
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 30
  }
}
```

### Azure OpenAI

```json
{
  "llm": {
    "provider": "azure_openai",
    "api_key": "${AZURE_OPENAI_API_KEY}",
    "api_base": "${AZURE_OPENAI_ENDPOINT}",
    "api_version": "2023-05-15",
    "deployment_name": "gpt-4"
  }
}
```

### Anthropic Claude

```json
{
  "llm": {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "api_key": "${ANTHROPIC_API_KEY}",
    "max_tokens": 2000
  }
}
```

## Extraction Configuration

### Entity and Relationship Extraction

```json
{
  "extraction": {
    "spacy_model": "en_core_web_sm",
    "enable_entity_extraction": true,
    "enable_relationship_extraction": true,
    "entity_types": [
      "PERSON",
      "ORG",
      "GPE",
      "DATE",
      "TIME",
      "MONEY",
      "PRODUCT"
    ],
    "custom_patterns": [
      {
        "label": "SKILL",
        "pattern": [{"LOWER": {"IN": ["python", "javascript", "sql"]}}]
      }
    ]
  }
}
```

### Advanced Extraction Settings

```json
{
  "extraction": {
    "use_llm_extraction": true,
    "llm_extraction_prompt": "Extract entities and relationships from: {text}",
    "confidence_threshold": 0.8,
    "max_entities_per_item": 20,
    "enable_coreference_resolution": true
  }
}
```

## Background Processing

### Basic Configuration

```json
{
  "background": {
    "enabled": true,
    "max_workers": 3,
    "queue_size": 1000,
    "batch_size": 10,
    "processing_interval": 5.0
  }
}
```

### Advanced Processing Options

```json
{
  "background": {
    "enabled": true,
    "max_workers": 5,
    "queue_size": 2000,
    "batch_size": 20,
    "processing_interval": 2.0,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "enable_priority_queue": true,
    "high_priority_types": ["episodic", "procedural"]
  }
}
```

## Similarity Metrics

### Weight Configuration

```json
{
  "similarity": {
    "semantic_weight": 0.4,
    "content_weight": 0.3,
    "temporal_weight": 0.2,
    "metadata_weight": 0.1,
    "enable_adaptive_weighting": true
  }
}
```

### Advanced Similarity Settings

```json
{
  "similarity": {
    "semantic_model": "all-MiniLM-L6-v2",
    "content_similarity_method": "jaccard",
    "temporal_decay_factor": 0.1,
    "metadata_fields": ["memory_type", "user_id", "tags"],
    "similarity_threshold": 0.3
  }
}
```

## Evolution Algorithms

### Algorithm Selection

```json
{
  "evolution": {
    "enabled": true,
    "algorithms": [
      "MaximalConnectivityEvolver",
      "RapidEnrichmentEvolver",
      "StrategicPruningEvolver",
      "HierarchicalOrganizationEvolver"
    ],
    "evolution_interval": 3600,
    "max_evolution_time": 300
  }
}
```

### Algorithm-Specific Settings

```json
{
  "evolution": {
    "MaximalConnectivityEvolver": {
      "connectivity_threshold": 0.5,
      "max_links_per_node": 10
    },
    "StrategicPruningEvolver": {
      "similarity_threshold": 0.8,
      "preserve_recent_days": 7,
      "archive_instead_of_delete": true
    },
    "RapidEnrichmentEvolver": {
      "enrichment_batch_size": 50,
      "enable_parallel_processing": true
    }
  }
}
```

## Ontology Configuration

### Basic Ontology Settings

```json
{
  "ontology": {
    "enabled": true,
    "storage_backend": "FileSystemOntologyStorage",
    "storage_path": "./ontologies",
    "default_ontology": "general_knowledge"
  }
}
```

### Advanced Ontology Management

```json
{
  "ontology": {
    "enabled": true,
    "auto_inference": true,
    "inference_threshold": 0.7,
    "enable_hitl_validation": true,
    "validation_rules": [
      "entity_type_consistency",
      "relationship_constraints",
      "domain_validation"
    ]
  }
}
```

## Performance Tuning

### High-Performance Configuration

```json
{
  "performance": {
    "enable_caching": true,
    "cache_size": 10000,
    "cache_ttl": 3600,
    "enable_batch_operations": true,
    "batch_size": 100,
    "connection_pool_size": 10,
    "query_timeout": 30
  }
}
```

### Memory Optimization

```json
{
  "memory_optimization": {
    "enable_lazy_loading": true,
    "max_memory_usage_mb": 2048,
    "garbage_collection_interval": 300,
    "enable_compression": true,
    "compression_algorithm": "gzip"
  }
}
```

## Environment-Specific Configurations

### Development Configuration

```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "graph_db": {
    "backend_class": "FalkorDBBackend",
    "host": "localhost",
    "port": 6379
  },
  "background": {
    "enabled": false
  },
  "evolution": {
    "enabled": false
  }
}
```

### Production Configuration

```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  "graph_db": {
    "backend_class": "FalkorDBBackend",
    "host": "${REDIS_HOST}",
    "port": 6379,
    "password": "${REDIS_PASSWORD}",
    "ssl": true
  },
  "background": {
    "enabled": true,
    "max_workers": 8
  },
  "performance": {
    "enable_caching": true,
    "connection_pool_size": 20
  }
}
```

## Configuration Loading

### Programmatic Configuration

```python
from smartmemory import SmartMemory
from smartmemory.configuration import MemoryConfig
from smartmemory.utils import get_config

# Load from file (recommended via environment variable SMARTMEMORY_CONFIG)
cfg = MemoryConfig(config_path="config.json")
cfg.validate()

# SmartMemory reads configuration via the configuration subsystem
memory = SmartMemory()

# Access configuration at runtime
vector_cfg = get_config('vector')
print(vector_cfg.get('backend'))
```

### Runtime Configuration Updates

```python
# Apply runtime config changes by editing the file, then either:
from smartmemory.configuration import MemoryConfig
cfg = MemoryConfig(config_path="config.json")
cfg.reload_if_stale(force=True)

# Or clear the cached config so subsequent get_config() calls reload
from smartmemory.utils import get_config, clear_config_cache
clear_config_cache()
current_config = get_config()
print(current_config.get('similarity'))
```

## Environment Variables

### Required Variables

```bash
# LLM API keys
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Database credentials
export REDIS_PASSWORD="your-redis-password"
export NEO4J_PASSWORD="your-neo4j-password"

# Vector store credentials
export PINECONE_API_KEY="your-pinecone-api-key"
```

### Optional Variables

```bash
# Custom configuration path
export SMARTMEMORY_CONFIG="/path/to/config.json"

# Environment specification
export SMARTMEMORY_ENV="production"

# Debug settings
export SMARTMEMORY_DEBUG="true"
export SMARTMEMORY_LOG_LEVEL="DEBUG"
```

## Configuration Validation

SmartMemory automatically validates configuration on startup:

```python
from smartmemory.configuration import MemoryConfig

# Validate configuration file
cfg = MemoryConfig(config_path="config.json")
cfg.validate()
```

## Best Practices

1. **Use environment variables** for sensitive information like API keys
2. **Separate configurations** for different environments (dev, staging, prod)
3. **Enable background processing** in production for better performance
4. **Configure appropriate worker counts** based on your hardware
5. **Use persistent storage** for graph and vector databases in production
6. **Enable caching** for frequently accessed data
7. **Monitor resource usage** and adjust configuration accordingly

## Troubleshooting

### Common Configuration Issues

1. **Invalid JSON syntax** - Use a JSON validator to check your configuration
2. **Missing environment variables** - Ensure all required variables are set
3. **Backend connection failures** - Verify database services are running
4. **Performance issues** - Adjust worker counts and batch sizes
5. **Memory usage** - Configure memory limits and garbage collection

### Configuration Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Validate and inspect configuration
from smartmemory.configuration import MemoryConfig
cfg = MemoryConfig(config_path="config.json")
cfg.validate()
print(cfg.graph_db)
```
