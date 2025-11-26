# System Architecture Overview

SmartMemory is built on a modular, component-based architecture that provides a unified interface for agentic memory operations while maintaining flexibility and extensibility. This document provides a comprehensive overview of the system architecture, design principles, and component interactions.

## Architecture Principles

### Core Design Principles

1. **Modularity**: Each component has a single responsibility and clear interfaces
2. **Extensibility**: New memory types, backends, and algorithms can be easily added
3. **Performance**: Optimized for both throughput and latency with background processing
4. **Reliability**: Robust error handling and graceful degradation
5. **Scalability**: Designed to handle large-scale memory operations
6. **Interoperability**: Standard interfaces for integration with external systems

### Architectural Patterns

- **Component-Based Architecture**: Loosely coupled components with dependency injection
- **Factory Pattern**: Consistent creation and initialization of memory components
- **Strategy Pattern**: Pluggable algorithms for similarity, evolution, and extraction
- **Observer Pattern**: Event-driven processing and monitoring
- **Adapter Pattern**: Integration with various backends and external services

## High-Level Architecture

```mermaid
graph TB
    subgraph "🌟 SmartMemory System Architecture"
        subgraph "API Layer"
            API["🚀 SmartMemory API"]
            CRUD["📝 CRUD Operations"]
            SEARCH["🔍 Search & Retrieval"]
            REL["🔗 Relationship Management"]
            BG["🌐 Background Processing"]
            
            API --> CRUD
            API --> SEARCH
            API --> REL
            API --> BG
        end
        
        subgraph "Core Components"
            subgraph "Memory Types"
                SEM["🔍 Semantic Memory"]
                EPI["📚 Episodic Memory"]
                PROC["⚙️ Procedural Memory"]
                WORK["💭 Working Memory"]
            end
            
            subgraph "Processing Pipeline"
                ING["📥 Ingestion Flow"]
                EXT["🔍 Entity Extraction"]
                ENR["📊 Enrichment"]
                EVO["🧬 Evolution Engine"]
            end
            
            subgraph "Graph Operations"
                GCRUD["💾 Graph CRUD"]
                LINK["🔗 Linking"]
                GSEARCH["🔍 Graph Search"]
                MON["📊 Monitoring"]
            end
            
            subgraph "External Integration"
                MCP["🔌 MCP Tools"]
                LLM["🤖 LLM Providers"]
                VEC["🗂️ Vector Stores"]
            end
        end
        
        subgraph "Storage Layer"
            subgraph "Graph Backend"
                FALKOR["🔴 FalkorDB"]
                NEO4J["🟢 Neo4j"]
            end
            
            subgraph "Vector Storage"
                CHROMA["🎨 ChromaDB"]
                PINECONE["🌲 Pinecone"]
                WEAVIATE["🕸️ Weaviate"]
            end
            
            subgraph "Configuration"
                CONFIG["⚙️ Config Manager"]
                FACTORY["🏭 Factories"]
                LOADER["📂 Config Loader"]
            end
        end
        
        subgraph "Infrastructure"
            DOCKER["🐳 Docker Services"]
            REDIS["🔴 Redis/FalkorDB"]
            CHROMADB["🎨 ChromaDB Service"]
            OPENAI["🤖 OpenAI API"]
        end
    end
    
    %% API Layer Connections
    CRUD --> SEM
    CRUD --> EPI
    CRUD --> PROC
    CRUD --> WORK
    
    SEARCH --> GSEARCH
    REL --> LINK
    BG --> EVO
    
    %% Processing Pipeline Connections
    ING --> EXT
    EXT --> ENR
    ENR --> EVO
    
    %% Storage Connections
    SEM --> GCRUD
    EPI --> GCRUD
    PROC --> GCRUD
    WORK --> GCRUD
    
    GCRUD --> FALKOR
    GCRUD --> NEO4J
    
    ENR --> CHROMA
    ENR --> PINECONE
    ENR --> WEAVIATE
    
    %% External Integration
    MCP --> API
    LLM --> ENR
    VEC --> CHROMA
    
    %% Infrastructure
    FALKOR --> REDIS
    CHROMA --> CHROMADB
    LLM --> OPENAI
    
    %% Configuration
    CONFIG --> FACTORY
    
    style API fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style GCRUD fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style EVO fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style DOCKER fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
  ```
  │  │   ├── Neo4j           │   ├── Pinecone                │
  ├─────────────────────────────────────────────────────────────┤
  │                Infrastructure Layer                        │
│  ├── Configuration       ├── Monitoring                   │
│  ├── Logging            ├── Error Handling                │
│  └── Background Processing                                 │
└─────────────────────────────────────────────────────────────┘

## Component Architecture

### SmartMemory Core Class

The `SmartMemory` class serves as the unified entry point and orchestrator:

```python
class SmartMemory(UnifiedMemoryBase):
    """
    Unified agentic memory store combining semantic, episodic, 
    procedural, and working memory.
    """
    
    def __init__(self, **kwargs):
        # Core storage
        self._graph = SmartGraph()
        
        # Component delegation
        self._crud = CRUD(self._graph)
        self._linking = Linking(self._graph)
        self._enrichment = Enrichment(self._graph)
        self._grounding = Grounding(self._graph)
        self._personalization = Personalization(self._graph)
        self._search = Search(self._graph)
        self._monitoring = Monitoring(self._graph)
        self._evolution = EvolutionOrchestrator(self)
        self._clustering = GlobalClustering(self._graph)
        
        # Temporal versioning
        self.version_tracker = VersionTracker(self._graph)
        
        # Processing flows
        self._ingestion_flow = MemoryIngestionFlow(self, ...)
        self.fast_ingestion_flow = FastIngestionFlow(self, ...)
        
        # Background processing
        self.background_processor = SimpleBackgroundProcessor()
```

**Key Responsibilities:**
- Unified API for all memory operations
- Component lifecycle management
- Request routing and delegation
- Background processing coordination
- Configuration management

### Memory Types

#### Semantic Memory
- **Purpose**: Long-term factual knowledge and concepts
- **Storage**: Graph nodes with rich semantic relationships
- **Processing**: Entity extraction, concept linking, ontology integration
- **Use Cases**: Knowledge bases, fact storage, concept hierarchies

#### Episodic Memory
- **Purpose**: Temporal experiences and events
- **Storage**: Time-ordered sequences with contextual metadata
- **Processing**: Temporal clustering, event correlation, narrative construction
- **Use Cases**: Conversation history, user interactions, experience tracking

#### Procedural Memory
- **Purpose**: Skills, procedures, and how-to knowledge
- **Storage**: Step-based workflows and process graphs
- **Processing**: Skill extraction, procedure optimization, workflow analysis
- **Use Cases**: Task automation, skill learning, process documentation

#### Working Memory
- **Purpose**: Active, short-term contextual information
- **Storage**: Temporary nodes with expiration policies
- **Processing**: Context management, relevance scoring, automatic cleanup
- **Use Cases**: Current conversation context, active task state

### Processing Pipeline

#### Ingestion Flow (11 Stages)
```
Input → Classification → Extraction → Storage → Linking → 
Vector → Enrichment → Grounding → Evolution → Clustering → Versioning
```

**Stages:**
1. **Input Adaptation**: Convert str/dict/MemoryItem to standard format
2. **Classification**: Determine memory type (semantic, episodic, procedural, working)
3. **Extraction**: Extract entities & relations (LLM → SpaCy → GLiNER → Relik fallback)
4. **Storage**: Create memory node + entity nodes in FalkorDB
5. **Linking**: Connect to related existing memories
6. **Vector Storage**: Generate embeddings, store in HNSW index
7. **Enrichment**: Add Wikipedia summaries, categories, metadata
8. **Grounding**: Create GROUNDED_IN edges to Wikipedia nodes
9. **Evolution**: Promote working → episodic/procedural if thresholds met
10. **Clustering**: SemHash + embedding deduplication of entities
11. **Versioning**: Create bi-temporal version record

#### Fast Ingestion Flow
```
Input → Quick Storage → Background Processing Queue
```

**Benefits:**
- Immediate storage for fast response times
- Background enrichment for full processing
- Scalable for high-throughput scenarios

### Graph Operations

#### CRUD Component
- **Create**: Add new memory items with validation
- **Read**: Retrieve items by ID, type, or criteria
- **Update**: Modify existing items with change tracking
- **Delete**: Remove items with cascade handling

#### Linking Component
- **Automatic Linking**: AI-driven relationship discovery
- **Explicit Linking**: User-defined relationships
- **Link Types**: Semantic, temporal, causal, hierarchical
- **Link Strength**: Weighted relationships with confidence scores

#### Search Component
- **Vector Search**: Semantic similarity using embeddings
- **Graph Traversal**: Relationship-based discovery
- **Hybrid Search**: Combined vector and graph approaches
- **Faceted Search**: Multi-dimensional filtering

### External Integration

#### MCP Tools
- **Standardized Interface**: Model Context Protocol compliance
- **Tool Discovery**: Auto-registration and service discovery
- **Agent Integration**: LangChain, AutoGen, custom frameworks
- **Operation Coverage**: Full CRUD and search capabilities

#### LLM Providers
- **OpenAI**: GPT models for extraction and enrichment
- **Anthropic**: Claude for analysis and reasoning
- **Azure OpenAI**: Enterprise-grade AI services
- **Local Models**: Support for self-hosted LLMs

#### Vector Stores
- **ChromaDB**: Local and cloud vector storage
- **Pinecone**: Managed vector database service
- **FAISS**: High-performance similarity search
- **Extensible**: Plugin architecture for new providers

## Data Flow Architecture

### Write Path
```
User Input → SmartMemory.add() → Ingestion Flow → 
Graph Storage + Vector Storage → Background Enrichment
```

### Read Path
```
Query → SmartMemory.search() → Search Component → 
Vector Search + Graph Traversal → Result Ranking → Response
```

### Background Processing
```
Queued Items → Background Processor → Enrichment Pipeline → 
Evolution Algorithms → Updated Storage
```

## Storage Architecture

### Graph Database Layer (FalkorDB)

**Node Types:**
- **Memory Nodes**: Core memory items with content and metadata
- **Entity Nodes**: Extracted entities with properties (dual-node pattern)
- **Wikipedia Nodes**: Global grounding nodes (shared across users)
- **Version Nodes**: Bi-temporal version records

**Relationship Types:**
- **CONTAINS**: Memory contains entity
- **GROUNDED_IN**: Entity grounded to Wikipedia
- **HAS_VERSION**: Memory has version history
- **RELATES_TO**: Semantic relationships
- **FOLLOWS**: Temporal sequences
- **PART_OF**: Hierarchical structures
- **SIMILAR_TO**: Similarity relationships

**Properties:**
- **Timestamps**: Creation, modification, access times
- **Scores**: Relevance, confidence, importance
- **Metadata**: User context, source information
- **Embeddings**: Vector representations for similarity

### Vector Storage Layer (FalkorDB HNSW)

**Unified Backend:**
- FalkorDB provides both graph and vector storage
- Native HNSW index with `vecf32` type
- Configurable parameters: M, efConstruction, efRuntime
- Cosine similarity search

**Embedding Types:**
- **Content Embeddings**: Full text semantic vectors (sentence-transformers)
- **Entity Embeddings**: Specific entity representations

**Indexing Strategy:**
- **HNSW Index**: Hierarchical Navigable Small World graphs
- **Tenant Isolation**: ScopeProvider filters all queries
- **Incremental Updates**: Real-time index maintenance

## Performance Architecture

### Optimization Strategies

#### Caching Layer
```python
# Multi-level caching strategy
L1_CACHE = LRUCache(maxsize=1000)      # Hot data
L2_CACHE = RedisCache(ttl=3600)        # Warm data
L3_CACHE = DiskCache(size="1GB")       # Cold data
```

#### Background Processing
```python
# Asynchronous processing pipeline
class BackgroundProcessor:
    def __init__(self, max_workers=3):
        self.enrichment_queue = Queue()
        self.evolution_queue = Queue()
        self.cleanup_queue = Queue()
```

#### Connection Pooling
```python
# Database connection management
class ConnectionPool:
    def __init__(self, backend_type, pool_size=10):
        self.pool = create_pool(backend_type, pool_size)
        self.health_checker = HealthChecker()
```

### Scalability Patterns

#### Horizontal Scaling
- **Sharded Storage**: Distribute data across multiple backends
- **Load Balancing**: Route requests across multiple instances
- **Microservices**: Decompose into specialized services

#### Vertical Scaling
- **Resource Optimization**: Memory and CPU tuning
- **Batch Processing**: Efficient bulk operations
- **Index Optimization**: Smart indexing strategies

## Security Architecture

### Access Control
- **User Isolation**: Strict user data separation
- **Permission Model**: Role-based access control
- **API Security**: Authentication and authorization
- **Data Encryption**: At-rest and in-transit protection

### Privacy Protection
- **Data Anonymization**: PII removal and masking
- **Retention Policies**: Automatic data expiration
- **Audit Logging**: Comprehensive access tracking
- **Compliance**: GDPR, CCPA, and other regulations

## Monitoring and Observability

### Metrics Collection
```python
# Key performance indicators
METRICS = {
    "throughput": "operations_per_second",
    "latency": "response_time_percentiles", 
    "accuracy": "search_relevance_scores",
    "resource_usage": "memory_cpu_disk_utilization"
}
```

### Health Monitoring
- **Component Health**: Individual component status
- **Dependency Health**: External service monitoring
- **Performance Alerts**: Threshold-based notifications
- **Capacity Planning**: Resource usage trends

### Debugging Support
- **Distributed Tracing**: Request flow tracking
- **Structured Logging**: Searchable log events
- **Error Tracking**: Exception monitoring and alerting
- **Performance Profiling**: Bottleneck identification

## Configuration Architecture

### Configuration Hierarchy
```
Environment Variables → Config Files → Runtime Parameters → Defaults
```

### Configuration Validation
```python
# Schema-based validation
CONFIG_SCHEMA = {
    "graph_db": {"backend_class": "FalkorDBBackend", "host": "localhost", ...}
    "vector": {"backend": "faiss", ...}
    "background": {"max_workers": 16, ...}
    "llm_provider": {"type": "string", "enum": ["openai", "anthropic", "azure"]},
    "background_processing": {"type": "boolean", "default": True}
}
```
{{ ... }}
- **Development**: Local backends, debug logging
- **Testing**: In-memory storage, mock services
- **Staging**: Production-like setup with test data
- **Production**: Optimized for performance and reliability

## Deployment Architecture

### Container Strategy
```dockerfile
# Multi-stage build for optimization
FROM python:3.12-slim as base
FROM base as dependencies
FROM dependencies as application
```

### Service Dependencies
```yaml
# Docker Compose orchestration
services:
  smartmemory:
    depends_on: [falkordb, chromadb, redis]
  falkordb:
    image: falkordb/falkordb:latest
  chromadb:
    image: chromadb/chroma:latest
  redis:
    image: redis:alpine
```

### Scaling Considerations
- **Stateless Design**: Horizontal scaling support
- **External State**: Database and cache separation
- **Health Checks**: Container orchestration support
- **Resource Limits**: Memory and CPU constraints

## Extension Points

### Custom Components
```python
# Plugin architecture
class CustomExtractor(BaseExtractor):
    def extract(self, content):
        # Custom extraction logic
        pass

# Registration
register_component("extractor", "custom", CustomExtractor)
```

### Custom Algorithms
```python
# Pluggable algorithms
class CustomSimilarity(BaseSimilarity):
    def compute_similarity(self, item1, item2):
        # Custom similarity computation
        pass

# Integration
similarity_registry.register("custom", CustomSimilarity)
```

### Custom Backends
```python
# Backend abstraction
class CustomBackend(BaseBackend):
    def connect(self):
        # Custom connection logic
        pass
    
    def query(self, cypher):
        # Custom query execution
        pass

# Registration
backend_registry.register("custom", CustomBackend)
```

## Future Architecture Considerations

### Planned Enhancements
- **Distributed Architecture**: Multi-node deployment
- **Event Sourcing**: Complete audit trail
- **CQRS Pattern**: Separate read/write models
- **GraphQL API**: Flexible query interface

### Research Directions
- **Neuromorphic Computing**: Brain-inspired architectures
- **Quantum Computing**: Quantum similarity algorithms
- **Edge Computing**: Distributed memory networks
- **Federated Learning**: Collaborative memory systems

## Best Practices

### Development Guidelines
1. **Component Isolation**: Minimize dependencies between components
2. **Interface Contracts**: Clear API definitions and contracts
3. **Error Handling**: Comprehensive error recovery strategies
4. **Testing Strategy**: Unit, integration, and performance tests
5. **Documentation**: Comprehensive API and architecture docs

### Operational Guidelines
1. **Monitoring**: Comprehensive observability setup
2. **Backup Strategy**: Regular data backup and recovery testing
3. **Capacity Planning**: Proactive resource management
4. **Security Updates**: Regular dependency and security updates
5. **Performance Tuning**: Continuous optimization based on metrics

This architecture provides a solid foundation for building scalable, reliable, and extensible agentic memory systems while maintaining flexibility for future enhancements and integrations.
