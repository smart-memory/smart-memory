# SmartMemory Documentation

> 📚 **Repository**: [smartmemory-ai/smart-memory](https://github.com/smart-memory/smart-memory)  
> 🐛 **Issues**: [Report bugs or request features](https://github.com/smart-memory/smart-memory/issues)  
> 📖 **Documentation**: [View online docs](https://smartmemory-ai.github.io/smart-memory/)

:::info
🧠 **SmartMemory** - Advanced Agentic Memory System for Intelligent Applications with Enterprise-Grade Multi-User Support
:::

---

## 🚀 What is SmartMemory?

SmartMemory is a **next-generation memory system** that goes far beyond simple storage and retrieval. It's designed specifically for **AI agents and intelligent applications** that need sophisticated memory capabilities with human-like cognitive processing, **complete user isolation**, and **intelligent assertion challenging**.

### 🌟 Key Features

- **🧠 Cognitive Memory Types**: Working, semantic, episodic, and procedural memory systems
- **🔄 Automatic Evolution**: Memories improve and consolidate over time
- **🎯 Intelligent Assertion Challenging**: LLM-based semantic analysis to detect and address contradictory statements
- **🔍 Semantic Search**: Find information by meaning, not just keywords
- **🌐 Framework Integration**: Works with LangChain, CrewAI, AutoGen, and more

---

## 🧠 **Memory Type Distribution**

SmartMemory organizes information into five distinct cognitive memory types, each optimized for different kinds of knowledge and usage patterns, mimicking human cognitive architecture.

```mermaid
flowchart TD
    A["🧠 Classified Memory"] --> B{"Memory Type?"}
    
    B -->|Facts & Knowledge| C["🔍 Semantic Memory"]
    B -->|Personal Events| D["📚 Episodic Memory"]
    B -->|Skills & Procedures| E["⚙️ Procedural Memory"]
    B -->|Active Context| F["💭 Working Memory"]
    B -->|Atomic Knowledge| G["🗂️ Zettelkasten Memory"]
    
    C --> H["🔗 Relationship Mapping"]
    D --> H
    E --> H
    G --> H
    F --> H
    
    H --> I["💾 Storage Layer"]
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style D fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style E fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

### 🧬 **Memory Evolution & Intelligence**

Memory improves itself in the background while you continue chatting.  Memories consolidate, prune, and enhance relationships as well as create new memories to self-improve.
The system also goes out to the web to find new information and ground existing memories in facts.

```mermaid
flowchart TB
    subgraph "Memory Lifecycle"
        A["📥 New Memory"] --> B["🔍 Initial Processing"]
        B --> C["💾 Storage"]
        C --> D["🔄 Background Evolution"]
        D --> E{"Evolution Type?"}
        
        E -->|Consolidate| F["🧬 Memory Consolidation"]
        E -->|Prune| G["✂️ Strategic Pruning"]
        E -->|Link| H["🔗 Relationship Enhancement"]
        E -->|Organize| I["📋 Hierarchical Organization"]
        
        F --> J["📊 Quality Assessment"]
        G --> J
        H --> J
        I --> J
        
        J --> K{"Quality Score?"}
        K -->|High| L["⭐ Promote Memory"]
        K -->|Medium| M["🔄 Continue Evolution"]
        K -->|Low| N["🗄️ Archive/Remove"]
        
        L --> O["🎯 Enhanced Retrieval"]
        M --> D
        N --> P["📋 Archive Storage"]
    end
    
    subgraph "Intelligence Features"
        Q["🤖 Agent Query"] --> R["🔍 Smart Search"]
        R --> S["🎯 Context Building"]
        S --> T["📊 Relevance Scoring"]
        T --> U["📤 Intelligent Response"]
        
        O --> R
        C --> R
    end
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style J fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style O fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style U fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

## 🗂️ **Zettelkasten Memory System**

SmartMemory includes a sophisticated Zettelkasten system for atomic knowledge management, automatically creating interconnected knowledge networks that evolve and strengthen over time.

### Zettelkasten Creation

```mermaid
flowchart LR
    A["📝 Knowledge Input"] --> B["🔍 Atomic Concept Extraction"]
    B --> C["📋 Zettel Creation"]
    C --> D["🏷️ Auto-Tagging"]
    D --> E["🔗 Link Discovery"]
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style E fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
```

### Knowledge Graph Evolution

```mermaid
flowchart LR
    F["🕸️ Concept Network"] --> G["📊 Hierarchical Clusters"]
    G --> H["🎯 Topic Modeling"]
    H --> I["🔍 Semantic Relationships"]
    I --> J["🧬 Link Strengthening"]
    J --> K["📈 Concept Merging"]
    K --> L["✂️ Redundancy Pruning"]
    L --> M["🌱 New Connection Discovery"]
    
    style F fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style M fill:#fff3e0,stroke:#f57c00,stroke-width:2px
```

### Intelligent Retrieval

```mermaid
flowchart LR
    N["🔍 Query Processing"] --> O["🎯 Relevance Scoring"]
    O --> P["📊 Context Assembly"]
    P --> Q["📤 Knowledge Response"]
    
    style N fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Q fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

## 🔍 **Advanced Search & Retrieval**

Multi-dimensional search capabilities that go beyond simple keyword matching, using semantic understanding, graph traversal, and temporal patterns to find the most relevant memories.

```mermaid
flowchart TD
    subgraph "Search Input"
        A["🔤 User Query"] --> B["🔍 Query Analysis"]
        B --> C["🎯 Intent Detection"]
        C --> D["📊 Context Extraction"]
    end
    
    subgraph "Search Strategies"
        D --> E["🔍 Semantic Search"]
        D --> F["🕸️ Graph Traversal"]
        D --> G["⏰ Temporal Search"]
        D --> H["🔗 Hybrid Search"]
        D --> I["📊 Similarity Search"]
    end
    
    subgraph "Memory Sources"
        E --> J["🔍 Semantic Memory"]
        F --> K["📚 Episodic Memory"]
        G --> L["⚙️ Procedural Memory"]
        H --> M["💭 Working Memory"]
        I --> N["🗂️ Zettelkasten Memory"]
    end
    
    subgraph "Result Processing"
        J --> O["📊 Relevance Scoring"]
        K --> O
        L --> O
        M --> O
        N --> O
        
        O --> P["🔗 Relationship Mapping"]
        P --> Q["📈 Ranking Algorithm"]
        Q --> R["📤 Contextual Results"]
    end
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style E fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style O fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style R fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

## 🎯 **Intelligent Assertion Challenging**

SmartMemory uses **LLM-based semantic analysis** to improve conversation quality by detecting and addressing contradictory statements:

```mermaid
flowchart TB
    subgraph "Assertion Analysis Pipeline"
        A["💬 User Statement"] --> B["🔍 Memory Retrieval"]
        B --> C["🧠 LLM Analysis"]
        C --> D{"Conflict Detected?"}
        
        D -->|No| E["✅ Normal Processing"]
        D -->|Yes| F["⚠️ Conflict Analysis"]
        
        F --> F1["📊 Confidence Scoring"]
        F --> F2["🎯 Conflict Type Classification"]
        F --> F3["💡 Suggested Response"]
        
        F1 --> G["🤝 Gentle Challenge"]
        F2 --> G
        F3 --> G
    end
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style G fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
```

### Assertion Challenging Features

- **🧠 Semantic Understanding**: Uses natural language processing instead of brittle regex patterns
- **📚 Contextual Memory Analysis**: Compares statements against stored conversation history
- **🤝 Gentle Challenging**: Respectfully questions inconsistencies to improve factual accuracy
- **📊 Confidence Scoring**: Provides nuanced assessment of potential conflicts
- **💡 Adaptive Responses**: Suggests appropriate ways to address contradictions

## 🔌 **MCP Server Integration**

Seamless integration with AI agents through the Model Context Protocol, providing standardized tools and interfaces for memory operations within agentic workflows.

```mermaid
flowchart TD
    subgraph "AI Agent Layer"
        A["🤖 AI Agent"] --> B["📡 MCP Client"]
        B --> C["🔌 Protocol Handler"]
    end
    
    subgraph "SmartMemory MCP Server"
        C --> D["📥 Request Router"]
        D --> E["🛠️ Tool Interface"]
        
        E --> F["💾 Memory Operations"]
        E --> G["🔍 Search Operations"]
        E --> H["🔗 Relationship Operations"]
        E --> I["📊 Analytics Operations"]
    end
    
    subgraph "Core Memory Engine"
        F --> J["🧠 Memory Manager"]
        G --> K["🔍 Search Engine"]
        H --> L["🕸️ Graph Engine"]
        I --> M["📈 Analytics Engine"]
    end
    
    subgraph "Storage Layer"
        J --> N["📊 Graph Database"]
        K --> O["🗂️ Vector Store"]
        L --> P["📋 Metadata Store"]
        M --> Q["📊 Analytics Store"]
    end
    
    subgraph "Background Processing"
        N --> R["🔄 Evolution Engine"]
        O --> R
        P --> R
        Q --> R
        
        R --> S["🧬 Memory Consolidation"]
        R --> T["✂️ Strategic Pruning"]
        R --> U["🔗 Link Enhancement"]
    end
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style E fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style J fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style R fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

## 💾 **Multi-Backend Storage Architecture**

Flexible storage architecture supporting multiple database backends, allowing you to choose the optimal storage solution for your specific performance and scalability requirements.

```mermaid
flowchart TD
    subgraph "Application Layer"
        A["🤖 SmartMemory API"] --> B["🔧 Storage Abstraction Layer"]
    end
    
    subgraph "Backend Selection"
        B --> C{"Storage Type?"}
        
        C -->|Graph| D["🕸️ Graph Backends"]
        C -->|Vector| E["🗂️ Vector Backends"]
        C -->|Metadata| F["📋 Metadata Backends"]
        C -->|Hybrid| G["🔄 Hybrid Backends"]
    end
    
    subgraph "Performance Features"
        D --> H["📈 Auto-Scaling"]
        E --> H
        F --> H
        G --> H
        
        H --> I["⚡ Caching Layer"]
        I --> J["🔄 Load Balancing"]
        J --> K["📊 Performance Monitoring"]
    end
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style C fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style H fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style K fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

## 🎯 **Agentic Framework Integration**

Seamless integration with major AI agent frameworks through specialized adapters, enabling SmartMemory to work natively within your existing agentic workflows and toolchains.

```mermaid
flowchart LR
    subgraph "AI Agent Frameworks"
        A["🦜 LangChain"]
        B["🦙 LlamaIndex"]
        C["🌾 Haystack"]
        D["🤖 AutoGen"]
        E["🚢 CrewAI"]
        F["🔧 Custom Agents"]
    end
    
    subgraph "SmartMemory Integration Layer"
        A --> G["🔌 LangChain Adapter"]
        B --> H["🔌 LlamaIndex Adapter"]
        C --> I["🔌 Haystack Adapter"]
        D --> J["🔌 AutoGen Adapter"]
        E --> K["🔌 CrewAI Adapter"]
        F --> L["🔌 Generic MCP Adapter"]
    end
    
    subgraph "Core SmartMemory"
        G --> M["🧠 Memory Engine"]
        H --> M
        I --> M
        J --> M
        K --> M
        L --> M
        
        M --> N["💾 Storage Layer"]
        M --> O["🔍 Search Engine"]
        M --> P["🧬 Evolution Engine"]
        M --> Q["📊 Analytics Engine"]
    end
    
    subgraph "Agent Capabilities"
        N --> R["💭 Memory Persistence"]
        O --> S["🔍 Intelligent Retrieval"]
        P --> T["🧬 Continuous Learning"]
        Q --> U["📈 Performance Insights"]
    end
    
    style A fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style G fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    style M fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style R fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

---

### 🎯 **Key Differentiators**

| Feature | Traditional Databases | SmartMemory |
|---------|----------------------|-------------|
| **Storage Architecture** | Static records | Hybrid triple storage (Graph + Vector + Metadata) |
| **Relationships** | Manual foreign keys | Automatic intelligent linking + evolution |
| **Processing** | Query-time only | Continuous background enrichment + evolution |
| **Intelligence** | None | Entity extraction, semantic analysis, ontology management |
| **Memory Types** | Single table/collection | 4 cognitive memory types (Semantic, Episodic, Procedural, Working) |
| **Agent Integration** | Basic CRUD | MCP protocol, tool interfaces, agentic workflows |
| **Evolution** | Static data | Memory consolidation, pruning, quality enhancement |
| **Ontology Management** | None | Hybrid freeform + structured knowledge extraction |
| **Human-in-the-Loop** | Manual queries only | HITL ontology editing, validation, and refinement |
| **Similarity Framework** | Basic text matching | Multi-dimensional similarity (semantic, temporal, structural) |
| **Background Processing** | None | Asynchronous enrichment, evolution, and optimization |
| **Context Awareness** | None | Temporal patterns, user context, cross-session intelligence |

## 🌟 Key Features & Architecture Blocks

### 🧠 **Cognitive Memory Types**

| Memory Type | Description | Key Features |
|-------------|-------------|-------------|
| **🔍 Semantic** | Facts, concepts, and general knowledge | Automatic concept extraction and linking, ontology-guided organization |
| **📚 Episodic** | Personal experiences and events | Rich temporal context, narrative understanding |
| **⚙️ Procedural** | Skills, procedures, and how-to knowledge | Step-by-step process understanding, action sequences |
| **💭 Working** | Temporary, active information processing | Context-aware short-term storage, dynamic attention management |
| **🗂️ Zettelkasten** | Atomic knowledge notes | Interconnected knowledge networks, automatic linking |

**Core Processing Capabilities:**
- **🤖 Automatic Entity Extraction**: Identify people, places, concepts, and relationships
- **🕸️ Intelligent Linking**: Discover connections between memories automatically
- **📈 Semantic Enrichment**: Enhance memories with contextual information
- **🎯 Similarity Analysis**: Multi-dimensional similarity scoring and clustering
- **🔄 Evolution Algorithms**: Continuous memory consolidation and optimization

### ⚡ **High-Performance Architecture**

:::tip Performance Features
**🚀 Key Performance Benefits:**
- **Asynchronous Processing**: Non-blocking memory enhancement
- **Fast Ingestion**: Store immediately, process in background
- **Multi-Backend Support**: Graph databases, vector stores, hybrid approaches
- **Intelligent Caching**: Smart retrieval optimization
- **Scalable Components**: Modular, extensible architecture
:::

### 🤖 **Agent-First Design**

:::note Built for AI Agents
- **🔌 MCP Integration**: Model Context Protocol support for seamless LLM integration
- **🛠️ Tool Interface**: Ready-to-use tools for agentic workflows
- **🧬 Evolution Algorithms**: Memory optimization designed specifically for AI agents
- **🎯 Contextual Retrieval**: Intelligent memory access patterns
## Quick Start

  ```python
  from smartmemory import SmartMemory
  
  # Initialize SmartMemory
  memory = SmartMemory()
  
  # Add memories
  memory.add("I learned Python programming in 2020")
  memory.add("Paris is the capital of France")
  memory.add("Attention mechanisms allow models to focus on relevant input parts", memory_type="zettel")
  memory.add("To make coffee: heat water, add grounds, brew for 4 minutes")
  
  # Search memories
  results = memory.search("programming languages")
  print(results)
  
  # Get related memories
  related = memory.get_neighbors(results[0].item_id)
  print(related)
  ```
|---------------------|-----------------|---------|----------|
| **Core Memory Types** |
| Semantic Memory | ✅ Advanced with entity extraction | ✅ Basic | ✅ Basic |
| Episodic Memory | ✅ Temporal + contextual | ❌ Limited | ✅ Basic |
| Procedural Memory | ✅ Skills + workflows | ❌ No | ❌ No |
| Working Memory | ✅ Adaptive capacity | ❌ No | ❌ No |
| Zettelkasten Memory | ✅ Atomic knowledge notes | ❌ No | ❌ No |
| **Intelligence & Processing** |
| Entity Extraction | ✅ Advanced NLP + LLM | ❌ Basic | ✅ Basic |
| Relationship Discovery | ✅ Automatic + intelligent | ❌ Manual | ✅ Limited |
| Grounding & Provenance | ✅ Full source attribution | ❌ No | ❌ Limited |
| Background Processing | ✅ Async + configurable | ❌ No | ✅ Basic |
| Evolution Algorithms | ✅ 14+ sophisticated evolvers | ❌ No | ❌ Basic |
| **Storage & Architecture** |
| Storage Architecture | ✅ Hybrid (Graph+Vector+Meta) | ❌ Vector only | ✅ Vector + basic |
| Graph Database Support | ✅ FalkorDB, Neo4j, Redis | ❌ No | ❌ Limited |
| Vector Database Support | ✅ ChromaDB, Pinecone, etc. | ✅ Multiple | ✅ Multiple |
| Multi-Backend Support | ✅ Pluggable backends | ❌ Limited | ✅ Yes |
| Hybrid Search | ✅ Semantic + Graph + Metadata | ❌ Vector only | ✅ Semantic |
| **AI Agent Integration** |
| MCP Protocol | ✅ Full MCP tool suite | ❌ No | ❌ Limited |
| LangChain Integration | ✅ Native support | ✅ Yes | ✅ Yes |
| AutoGen Compatible | ✅ Multi-agent support | ❌ Limited | ✅ Basic |
| Custom Tool Creation | ✅ Extensible toolbox | ❌ Limited | ✅ Basic |
| Agent Memory Isolation | ✅ Multi-tenant + namespaces | ❌ Basic | ✅ User-based |
| **Search & Retrieval** |
| Semantic Search | ✅ Advanced embeddings | ✅ Yes | ✅ Yes |
| Graph Traversal | ✅ Relationship-based discovery | ❌ No | ❌ Limited |
| Multi-Modal Queries | ✅ Content+Meta+Temporal | ❌ Basic | ✅ Content |
| Context-Aware Search | ✅ User+Time+Type filtering | ❌ Limited | ✅ Basic |
| Real-Time Performance | ✅ Sub-millisecond | ✅ Fast | ✅ Fast |
| **Developer Experience** |
| API Completeness | ✅ Full CRUD + advanced ops | ✅ Basic CRUD | ✅ CRUD + search |
| Configuration System | ✅ JSON/YAML + env vars | ❌ Basic | ✅ Config files |
| Documentation Quality | ✅ Comprehensive + examples | ❌ Limited | ✅ Good |
| Testing Framework | ✅ Full test suite + benchmarks | ❌ Basic | ✅ Tests |
| Monitoring & Analytics | ✅ Metrics + health checks | ❌ No | ❌ Basic |
| **Scalability & Performance** |
| Horizontal Scaling | ✅ Multi-worker + load balancing | ❌ Limited | ✅ Cloud-native |
| Caching Strategy | ✅ Multi-level intelligent caching | ❌ Basic | ✅ Basic |
| Batch Processing | ✅ Efficient bulk operations | ❌ No | ✅ Limited |
| Background Evolution | ✅ Continuous optimization | ❌ No | ❌ No |
| Resource Management | ✅ Connection pooling + optimization | ❌ Basic | ✅ Cloud-managed |
| **Enterprise Features** |
| Multi-Tenancy | ✅ Secure namespace isolation | ❌ No | ✅ User isolation |
| Audit Trails | ✅ Full provenance tracking | ❌ No | ❌ Limited |
| Security Model | ✅ Comprehensive access control | ❌ Basic | ✅ User-based |
| Backup & Recovery | ✅ Point-in-time recovery | ❌ Manual | ✅ Cloud backup |
| Compliance Support | ✅ GDPR + audit ready | ❌ No | ✅ Basic |

### **Key Differentiators**

🧠 **SmartMemory**: Most comprehensive system with 5 memory types, advanced evolution algorithms, hybrid storage, and enterprise-grade features

🔧 **Zep**: Lightweight vector-based system focused on simplicity and basic retrieval

☁️ **Mem0**: Cloud-native solution with good LLM integration but limited memory sophistication

### **Best Use Cases**

- **SmartMemory**: Complex AI agents, enterprise applications, research systems, sophisticated conversational AI
- **Zep**: Simple chatbots, basic RAG applications, proof-of-concepts
- **Mem0**: Cloud-first applications, basic personalization, simple memory needs

## Use Cases

- **Conversational AI**: Persistent memory for chatbots and assistants
- **Learning Systems**: Educational applications with knowledge tracking
- **Knowledge Management**: Enterprise knowledge bases with intelligent organization
- **Research Tools**: Academic and scientific knowledge organization
- **Personal AI**: Individual memory systems for personal assistants

## Next Steps

1. **[Installation](getting-started/installation)** - Set up SmartMemory in your environment
2. **[Quick Start](getting-started/quick-start)** - Build your first memory-enabled application
3. **[Core Concepts](concepts/overview)** - Understand the fundamental concepts
4. **[API Reference](api/smart-memory)** - Detailed API documentation
---

*SmartMemory is designed for developers building intelligent applications that need sophisticated memory capabilities. Whether you're creating conversational AI, learning systems, or knowledge management tools, SmartMemory provides the foundation for intelligent memory processing.*
