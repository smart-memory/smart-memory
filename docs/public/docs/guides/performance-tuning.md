# Performance Tuning

This guide covers optimization strategies and best practices for maximizing SmartMemory performance in production environments.

## Performance Overview

SmartMemory performance depends on several key factors:
- **Graph database backend** (FalkorDB, Neo4j, In-Memory)
- **Vector store configuration** (ChromaDB, Pinecone, FAISS)
- **Background processing settings**
- **Caching strategies**
- **Query optimization**

## Backend Optimization

### Graph Database Tuning

#### FalkorDB Configuration
```python
falkordb_config = {
    "graph_backend": {
        "type": "falkordb",
        "host": "localhost",
        "port": 6379,
        "connection_pool": {
            "max_connections": 20,
            "retry_on_timeout": True,
            "socket_keepalive": True,
            "socket_keepalive_options": {
                "TCP_KEEPIDLE": 1,
                "TCP_KEEPINTVL": 3,
                "TCP_KEEPCNT": 5
            }
        },
        "query_optimization": {
            "enable_query_cache": True,
            "cache_size": "256MB",
            "query_timeout": 30
        }
    }
}
```

#### Neo4j Configuration
```python
neo4j_config = {
    "graph_backend": {
        "type": "neo4j",
        "uri": "bolt://localhost:7687",
        "connection_pool": {
            "max_connection_lifetime": 3600,
            "max_connection_pool_size": 50,
            "connection_acquisition_timeout": 60,
            "connection_timeout": 30
        },
        "performance": {
            "fetch_size": 1000,
            "max_retry_time": 30,
            "initial_retry_delay": 1.0,
            "multiplier": 2.0,
            "jitter_factor": 0.2
        }
    }
}
```

### Vector Store Optimization

#### ChromaDB Performance
```python
chromadb_config = {
    "vector_store": {
        "type": "chromadb",
        "persist_directory": "./chroma_db",
        "settings": {
            "chroma_db_impl": "duckdb+parquet",
            "persist_directory": "./chroma_db",
            "anonymized_telemetry": False
        },
        "performance": {
            "batch_size": 1000,
            "max_batch_size": 5000,
            "embedding_cache_size": 10000,
            "query_cache_size": 1000
        }
    }
}
```

#### Pinecone Optimization
```python
pinecone_config = {
    "vector_store": {
        "type": "pinecone",
        "api_key": "${PINECONE_API_KEY}",
        "environment": "us-west1-gcp",
        "index_config": {
            "metric": "cosine",
            "pods": 1,
            "replicas": 1,
            "pod_type": "p1.x1"
        },
        "performance": {
            "batch_size": 100,
            "max_retries": 3,
            "request_timeout": 30,
            "upsert_batch_size": 1000
        }
    }
}
```

## Memory Configuration Optimization

### Background Processing Tuning

```python
background_config = {
    "background_processing": {
        "enabled": True,
        "max_workers": 4,  # Adjust based on CPU cores
        "queue_size": 10000,
        "batch_processing": {
            "enabled": True,
            "batch_size": 100,
            "batch_timeout": 5.0
        },
        "worker_config": {
            "enrichment_workers": 2,
            "evolution_workers": 1,
            "cleanup_workers": 1
        }
    }
}
```

### Caching Configuration

```python
caching_config = {
    "caching": {
        "l1_cache": {
            "enabled": True,
            "max_size": 1000,
            "ttl": 300  # 5 minutes
        },
        "l2_cache": {
            "enabled": True,
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "max_memory": "1GB",
            "ttl": 3600  # 1 hour
        },
        "query_cache": {
            "enabled": True,
            "max_size": 500,
            "ttl": 600  # 10 minutes
        }
    }
}
```

## Query Optimization

### Efficient Search Patterns

#### Optimized Vector Search
```python
# Use specific memory types to reduce search space
results = memory.search(
    query="machine learning",
    memory_type="semantic",  # Limit to specific type
    top_k=10,  # Reasonable limit
    filters={
        "user_id": "user123",  # Pre-filter by user
        "date_range": {
            "start": "2024-01-01",
            "end": "2024-12-31"
        }
    }
)
# Batch multiple searches
queries = ["AI", "ML", "Deep Learning"]
results = {q: memory.search(q, top_k=10, memory_type="semantic") for q in queries}
```
  ```python
  # Get direct neighbors of a node
  related = memory.get_neighbors("memory_123")
  # Apply any filtering/weighting client-side as needed
  ```
### Bulk Operations

#### Batch Ingestion

```python
# Efficient batch processing
items = [
    {"content": f"Document {i}", "metadata": {"batch": "1"}}
    for i in range(1000)
]

# Process in optimized batches (enqueue for background)
def chunk(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

for chunk_items in chunk(items, 100):
    for it in chunk_items:
        memory.ingest(it, sync=False)
```

#### Bulk Updates
```python
# Batch update operations
updates = [
    {"item_id": f"memory_{i}", "properties": {"importance": 0.8}}
    for i in range(100)
]

for upd in updates:
    memory.update_properties(upd["item_id"], upd["properties"])  # merge semantics
```

## Monitoring and Profiling

### Performance Metrics Collection

```python
from smartmemory.monitoring import PerformanceMonitor

# Initialize performance monitoring
monitor = PerformanceMonitor(
    enable_query_profiling=True,
    enable_resource_monitoring=True,
    sample_rate=0.1  # Sample 10% of operations
)

# Monitor specific operations
@monitor.profile_operation("custom_search")
def complex_search(query):
    return memory.search(
        query,
        top_k=50,
        include_relationships=True
    )

# Get performance statistics
stats = monitor.get_performance_stats()
print(f"Average query time: {stats['avg_query_time']:.2f}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
```

### Resource Monitoring

```python
import smartmemory.utils
import psutil
import asyncio


class ResourceMonitor:
    def __init__(self, memory_instance):
        self.memory = memory_instance
        self.metrics = []

    async def monitor_resources(self):
        while True:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()

            # Optional SmartMemory summary (counts by type)
            summary = self.memory.summary()

            metrics = {
                "timestamp": smartmemory.utils.now(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory_info.percent,
                # add SmartMemory-derived metrics as needed from `summary`
            }

            self.metrics.append(metrics)

            # Alert on high resources usage
            if cpu_percent > 80 or memory_info.percent > 85:
                await self.send_alert(metrics)

            await asyncio.sleep(60)  # Check every minute
```

## Optimization Strategies

### Memory Lifecycle Optimization

```python
# Configure efficient memory lifecycle
lifecycle_config = {
    "retention_policies": {
        "working_memory": {
            "max_items": 1000,
            "cleanup_interval": 300,  # 5 minutes
            "cleanup_strategy": "lru"
        },
        "episodic_memory": {
            "max_age_days": 90,
            "importance_threshold": 0.3,
            "cleanup_interval": 3600  # 1 hour
        }
    },
    "optimization": {
        "enable_compression": True,
        "compression_threshold": 1000,  # Compress after 1000 characters
        "enable_indexing": True,
        "index_rebuild_interval": 86400  # 24 hours
    }
}

memory.configure_lifecycle(lifecycle_config)
```

### Query Pattern Optimization

```python
class QueryOptimizer:
    def __init__(self, memory):
        self.memory = memory
        self.query_patterns = {}
    
    def optimize_query(self, query, user_context=None):
        # Analyze query pattern
        pattern = self.analyze_query_pattern(query)
        
        # Apply optimizations based on pattern
        if pattern == "semantic_search":
            return self.optimize_semantic_search(query)
        elif pattern == "temporal_search":
            return self.optimize_temporal_search(query, user_context)
        elif pattern == "relationship_search":
            return self.optimize_relationship_search(query)
        
        return query
    
    def optimize_semantic_search(self, query):
        # Use semantic-specific optimizations
        return {
            "query": query,
            "memory_type": "semantic",
            "use_vector_index": True,
            "embedding_cache": True
        }
    
    def optimize_temporal_search(self, query, user_context):
        # Add temporal constraints
        time_range = self.infer_time_range(query, user_context)
        return {
            "query": query,
            "memory_type": "episodic",
            "time_range": time_range,
            "use_temporal_index": True
        }
```

## Scaling Strategies

### Horizontal Scaling

```python
# Distributed SmartMemory setup
class DistributedSmartMemory:
    def __init__(self, shard_config):
        self.shards = {}
        self.load_balancer = LoadBalancer()
        
        for shard_id, config in shard_config.items():
            self.shards[shard_id] = SmartMemory(
                config=config,
                shard_id=shard_id
            )
    
    def route_query(self, query, user_id=None):
        # Route based on user_id or content hash
        if user_id:
            shard_id = self.hash_user_to_shard(user_id)
        else:
            shard_id = self.hash_content_to_shard(query)
        
        return self.shards[shard_id]
    
    async def distributed_search(self, query, user_id=None):
        if user_id:
            # Search specific user shard
            shard = self.route_query(query, user_id)
            return await shard.search_async(query)
        else:
            # Search all shards and merge results
            tasks = [
                shard.search_async(query)
                for shard in self.shards.values()
            ]
            
            all_results = await asyncio.gather(*tasks)
            return self.merge_distributed_results(all_results)
```

### Vertical Scaling

```python
# High-performance single-instance configuration
high_performance_config = {
    "graph_backend": {
        "type": "falkordb",
        "connection_pool_size": 50,
        "query_cache_size": "1GB",
        "memory_mapping": True
    },
    "vector_store": {
        "type": "faiss",
        "index_type": "IVF",
        "nlist": 1024,
        "nprobe": 64,
        "use_gpu": True  # If available
    },
    "background_processing": {
        "max_workers": 16,
        "queue_size": 50000,
        "batch_size": 500
    },
    "caching": {
        "memory_limit": "8GB",
        "enable_compression": True,
        "compression_algorithm": "lz4"
    }
}
```

## Performance Testing

### Benchmarking Framework

```python
import smartmemory.utils
import time
import statistics
from concurrent.futures import ThreadPoolExecutor


class PerformanceBenchmark:
    def __init__(self, memory):
        self.memory = memory
        self.results = {}

    def benchmark_search_performance(self, queries, iterations=100):
        """Benchmark search performance with multiple queries."""
        times = []

        for _ in range(iterations):
            start_time = time.time()

            for query in queries:
                results = self.memory.search(query, top_k=10)

            end_time = time.time()
            times.append(end_time - start_time)

        return {
            "avg_time": statistics.mean(times),
            "median_time": statistics.median(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_dev": statistics.stdev(times)
        }

    def benchmark_concurrent_access(self, query, concurrent_users=10):
        """Benchmark concurrent access performance."""

        def search_worker():
            start_time = time.time()
            results = self.memory.search(query)
            end_time = time.time()
            return end_time - start_time

        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(search_worker)
                for _ in range(concurrent_users)
            ]

            times = [future.result() for future in futures]

        return {
            "concurrent_users": concurrent_users,
            "avg_response_time": statistics.mean(times),
            "max_response_time": max(times),
            "throughput": concurrent_users / max(times)
        }

    def run_full_benchmark(self):
        """Run comprehensive performance benchmark."""
        test_queries = [
            "machine learning algorithms",
            "data science projects",
            "artificial intelligence research",
            "software development practices"
        ]

        # Search performance
        search_perf = self.benchmark_search_performance(test_queries)

        # Concurrent access
        concurrent_perf = self.benchmark_concurrent_access(
            "performance test",
            concurrent_users=20
        )

        # Memory operations
        add_perf = self.benchmark_add_performance()

        return {
            "search_performance": search_perf,
            "concurrent_performance": concurrent_perf,
            "add_performance": add_perf,
            "timestamp": smartmemory.utils.now()
        }
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### Slow Search Queries
```python
# Use built-in debug facility
debug = memory.debug_search("diagnostic query", top_k=10)
print("Graph backend:", debug.get("graph_backend"))
print("Graph results:", debug.get("graph_search_count"))
print("Search component:", debug.get("search_component_count"))
```

#### Memory Leaks
```python
# Monitor memory usage
def monitor_memory_usage(memory):
    import tracemalloc
    
    tracemalloc.start()
    
    # Perform operations
    for i in range(1000):
        memory.add(f"Test content {i}")
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "current_memory": current / 1024 / 1024,  # MB
        "peak_memory": peak / 1024 / 1024,  # MB
        "memory_per_item": current / 1000  # bytes per item
    }
```

### Performance Optimization Checklist

1. **Backend Configuration**
   - [ ] Optimize connection pool sizes
   - [ ] Enable query caching
   - [ ] Configure appropriate timeouts
   - [ ] Use connection keep-alive

2. **Vector Store Optimization**
   - [ ] Choose appropriate index type
   - [ ] Optimize batch sizes
   - [ ] Enable embedding caching
   - [ ] Use GPU acceleration if available

3. **Query Optimization**
   - [ ] Use specific memory types
   - [ ] Apply pre-filters
   - [ ] Limit result counts
   - [ ] Cache frequent queries

4. **Background Processing**
   - [ ] Tune worker counts
   - [ ] Optimize batch sizes
   - [ ] Configure queue sizes
   - [ ] Monitor queue depths

5. **Monitoring Setup**
   - [ ] Enable performance metrics
   - [ ] Set up alerting
   - [ ] Monitor resource usage
   - [ ] Track query patterns

## Best Practices Summary

1. **Start with profiling** to identify bottlenecks
2. **Optimize the biggest bottlenecks first**
3. **Use appropriate data structures** for your access patterns
4. **Implement caching strategically**
5. **Monitor continuously** and adjust based on real usage
6. **Test performance changes** with realistic workloads
7. **Plan for scaling** before you need it

Performance tuning is an iterative process. Start with these optimizations and continuously monitor and adjust based on your specific usage patterns and requirements.
