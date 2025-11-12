# Installation

> 📚 **Repository**: [smartmemory-ai/smart-memory](https://github.com/smart-memory/smart-memory)  
> 🐛 **Issues**: [Report bugs or request features](https://github.com/smart-memory/smart-memory/issues)

*Note, skip this guide for now, it's not public yet.*

This guide will help you install SmartMemory and its dependencies in your development environment.

## 🚀 Quick Start with Docker (Recommended)

The fastest way to get SmartMemory running is with Docker, which handles all backend services automatically.

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ for the SmartMemory package

### 1. Clone and Setup (Not public yet, sorry)

```bash
# Clone the repository
git clone https://github.com/regression-io/smart-memory.git
# Install SmartMemory package
pip install -e .

### 2. Start Backend Services

```bash
# Start required services (Redis/FalkorDB)
docker-compose up -d

# Verify services are running
docker-compose ps
```

### 3. Verify Installation

```python
from smartmemory import SmartMemory

# Initialize with default Docker configuration
memory = SmartMemory()

# Test basic functionality
memory.add("Hello, SmartMemory!")
results = memory.search("Hello")
print(f"Found {len(results)} memories")
```

**✅ You're ready to go!** Skip to the [Quick Start Guide](./quick-start.md) to begin using SmartMemory.

---

## Manual Installation

### Requirements

- Python 3.8 or higher
- pip or conda package manager
- Optional: Docker (for graph database backends)

## Basic Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/regression-io/smart-memory.git
cd smart-memory

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Using conda

```bash
# Clone the repository
git clone https://github.com/regression-io/smart-memory.git
cd smart-memory

# Create conda environment
conda create -n smartmemory python=3.9
conda activate smartmemory

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Dependencies

SmartMemory requires the following core dependencies:

```txt
# Core dependencies
pydantic>=2.0.0
numpy>=1.21.0
sentence-transformers>=2.2.0
spacy>=3.4.0
openai>=1.0.0

# Graph database backends
falkordb>=1.0.0  # Recommended
neo4j>=5.0.0     # Alternative

# Vector database
chromadb>=0.4.0

# Background processing
asyncio
concurrent.futures

# Development tools
pytest>=7.0.0
black>=22.0.0
```

## Backend Setup

SmartMemory supports multiple backend configurations. Choose the one that best fits your needs.

### Option 1: FalkorDB Backend (Recommended)

FalkorDB provides high-performance graph operations with Redis compatibility.

#### Using Docker (Recommended)

```bash
# Start FalkorDB with Docker
docker run -p 6379:6379 falkordb/falkordb:latest

# Or use docker-compose
docker-compose -f docker-compose.services.yml up falkordb
```

#### Manual Installation

```bash
# Install Redis with FalkorDB module
# Follow instructions at: https://docs.falkordb.com/
```

### Option 2: Neo4j Backend

```bash
# Using Docker
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v $HOME/neo4j/data:/data \
    -v $HOME/neo4j/logs:/logs \
    -v $HOME/neo4j/import:/var/lib/neo4j/import \
    -v $HOME/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/password \
    neo4j:latest
```



## Configuration

Create a configuration file to customize SmartMemory behavior:

### config.json

```json
{
  "graph": {
    "backend": "FalkorDBBackend",
    "host": "localhost",
    "port": 6379,
    "graph_name": "smartmemory"
  },
  "vector_store": {
    "backend": "ChromaDBBackend",
    "persist_directory": "./chroma_db"
  },
  "llm": {
    "provider": "openai",
    "model": "gpt-4",
    "api_key": "${OPENAI_API_KEY}"
  },
  "extraction": {
    "spacy_model": "en_core_web_sm"
  },
  "background_processing": {
    "enabled": true,
    "max_workers": 3
  }
}
```

### Environment Variables

Set up required environment variables:

```bash
# OpenAI API key for LLM operations
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Custom config path
export SMARTMEMORY_CONFIG_PATH="/path/to/your/config.json"
```

## Verification

Verify your installation with a simple test:

```python
from smartmemory import SmartMemory

# Initialize SmartMemory
memory = SmartMemory()

# Test basic functionality
memory.add("Test memory item")
results = memory.search("test")

print(f"Installation successful! Found {len(results)} results.")
```

## Troubleshooting

### Common Issues

#### 1. spaCy Model Not Found

```bash
# Download the English language models
python -m spacy download en_core_web_sm
```

#### 2. FalkorDB Connection Error

```bash
# Check if FalkorDB is running
docker ps | grep falkordb

# Check connection
redis-cli -p 6379 ping
```

#### 3. ChromaDB Permission Error

```bash
# Fix permissions for ChromaDB directory
chmod -R 755 ./chroma_db
```

#### 4. OpenAI API Key Error

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

### Performance Optimization

For production deployments:

1. **Use persistent storage** for graph and vector databases
2. **Enable background processing** for better performance
3. **Configure appropriate worker counts** based on your hardware
4. **Use GPU acceleration** for sentence transformers if available

```python
# Example production configuration
memory = SmartMemory(config_path="production_config.json")
```

## Next Steps

- [Quick Start Guide](quick-start) - Build your first application
- [Configuration Guide](configuration) - Detailed configuration options
- [Basic Usage](../guides/basic-usage) - Learn the core APIs
