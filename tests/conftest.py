"""
Test configuration and fixtures for SmartMemory test suite.
Provides common fixtures, test data, and setup/teardown functionality.
"""
import pytest
import tempfile
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional
from unittest.mock import Mock, patch
import os

from smartmemory.smart_memory import SmartMemory
from smartmemory.models.memory_item import MemoryItem
from smartmemory.conversation.manager import ConversationManager
from smartmemory.conversation.context import ConversationContext


@pytest.fixture(scope="session")
def test_config():
    """Test configuration with isolated backends for unit tests."""
    return {
        "graph_db": {
            "backend_class": "FalkorDBBackend",
            "host": "localhost",
            "port": 6379,
            "database": "test_smartmemory"
        },
        "vector_store": {
            "backend": "chromadb",
            "persist_directory": tempfile.mkdtemp(prefix="test_chroma_"),
            "collection_name": "test_collection"
        },
        "cache": {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 15  # Use separate test database
            }
        }
    }


@pytest.fixture(scope="session")
def integration_config():
    """Real configuration for integration tests - no mocks."""
    return {
        "graph_db": {
            "backend_class": "FalkorDBBackend",
            "host": "localhost",
            "port": 9010,
            "database": "integration_test_smartmemory"
        },
        "vector_store": {
            "backend": "chromadb",
            "persist_directory": "./test_data/integration_chroma",
            "collection_name": "integration_test_collection"
        },
        "cache": {
            "redis": {
                "host": "localhost",
                "port": 9012,
                "db": 14  # Separate integration test database
            }
        },
        "extractors": {
            "openai": {
                "api_key": os.getenv("OPENAI_API_KEY", "test-key"),
                "model": "gpt-3.5-turbo"
            }
        },
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-ada-002"
        }
    }


@pytest.fixture
def real_smartmemory_for_integration(integration_config):
    """Real SmartMemory instance for integration tests - NO MOCKS."""
    # Import here to avoid circular imports
    from smartmemory.smart_memory import SmartMemory

    # Set environment variables to match integration config/docker services
    os.environ["FALKORDB_PORT"] = str(integration_config["graph_db"]["port"])
    os.environ["REDIS_PORT"] = str(integration_config["cache"]["redis"]["port"])
    os.environ["VECTOR_BACKEND"] = "falkordb" # Use FalkorDB as default if Chroma is missing
    
    # Try to create real SmartMemory instance; skip gracefully if backends are unavailable
    try:
        memory = SmartMemory()
    except Exception as e:
        import pytest as _pytest
        _pytest.skip(f"Integration environment not ready: {e}")

    yield memory

    # Cleanup after integration tests
    try:
        memory.clear()  # Clear test data
    except Exception:
        pass  # Ignore cleanup errors

    # Reset environment
    if 'SMARTMEMORY_ENV' in os.environ:
        del os.environ['SMARTMEMORY_ENV']
    
    # Cleanup env vars
    os.environ.pop("FALKORDB_PORT", None)
    os.environ.pop("REDIS_PORT", None)
    os.environ.pop("VECTOR_BACKEND", None)


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="smartmemory_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_memory_items():
    """Sample memory items for testing."""
    return [
        MemoryItem(
            content="Python is a programming language",
            memory_type="semantic",
            user_id="user1",
            metadata={"source": "test", "confidence": 0.9}
        ),
        MemoryItem(
            content="I learned Python yesterday",
            memory_type="episodic",
            user_id="user1",
            metadata={"timestamp": datetime.now(timezone.utc).isoformat()}
        ),
        MemoryItem(
            content="How to write a function in Python",
            memory_type="procedural",
            user_id="user1",
            metadata={"steps": ["def", "parameters", "return"]}
        ),
        MemoryItem(
            content="Currently working on memory tests",
            memory_type="working",
            user_id="user1",
            metadata={"priority": "high"}
        )
    ]


@pytest.fixture
def sample_entities():
    """Sample entities for testing."""
    return [
        {"name": "Python", "type": "PROGRAMMING_LANGUAGE", "confidence": 0.95},
        {"name": "function", "type": "CONCEPT", "confidence": 0.85},
        {"name": "user1", "type": "PERSON", "confidence": 1.0}
    ]


@pytest.fixture
def sample_relations():
    """Sample relations for testing."""
    return [
        {"subject": "Python", "predicate": "IS_A", "object": "programming language"},
        {"subject": "function", "predicate": "PART_OF", "object": "Python"},
        {"subject": "user1", "predicate": "LEARNED", "object": "Python"}
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dim embedding
        [0.2, 0.3, 0.4, 0.5, 0.6] * 100,
        [0.3, 0.4, 0.5, 0.6, 0.7] * 100,
        [0.4, 0.5, 0.6, 0.7, 0.8] * 100
    ]


@pytest.fixture
def conversation_context():
    """Sample conversation context."""
    return ConversationContext(
        conversation_id="test_conv_1",
        user_id="user1",
        metadata={"topic": "testing", "session_start": datetime.now(timezone.utc).isoformat()}
    )


@pytest.fixture
def conversation_manager():
    """Conversation manager instance."""
    return ConversationManager()


@pytest.fixture
def mock_smartmemory_dependencies():
    """Mock all SmartMemory dependencies for isolated testing."""
    with patch('smartmemory.configuration.get_config') as mock_get_config, \
         patch('smartmemory.stores.vector.vector_store.VectorStore') as mock_vector_store, \
         patch('smartmemory.smart_memory.SmartGraph') as mock_smart_graph, \
         patch('smartmemory.smart_memory.GraphOperations') as mock_graph_ops, \
         patch('smartmemory.smart_memory.CRUD') as mock_crud, \
         patch('smartmemory.smart_memory.Linking') as mock_linking, \
         patch('smartmemory.smart_memory.Enrichment') as mock_enrichment, \
         patch('smartmemory.smart_memory.Grounding') as mock_grounding, \
         patch('smartmemory.smart_memory.Personalization') as mock_personalization, \
         patch('smartmemory.smart_memory.Search') as mock_search, \
         patch('smartmemory.smart_memory.Monitoring') as mock_monitoring, \
         patch('smartmemory.smart_memory.EvolutionOrchestrator') as mock_evolution, \
         patch('smartmemory.smart_memory.ExternalResolver') as mock_external_resolver, \
         patch('smartmemory.smart_memory.MemoryIngestionFlow') as mock_ingestion_flow:
        
        # Mock configuration with proper object structure
        mock_config = Mock()
        mock_config.host = 'localhost'
        mock_config.port = 6379
        mock_config.backend = 'falkordb'
        mock_get_config.return_value = mock_config
        
        # Mock VectorStore with proper method signatures
        mock_vector_instance = Mock()
        mock_vector_instance.add.return_value = Mock()
        mock_vector_instance.search.return_value = []
        mock_vector_instance.delete.return_value = Mock()
        mock_vector_instance.clear.return_value = Mock()
        mock_vector_store.get.return_value = mock_vector_instance
        
        # Configure mock instances
        mock_graph_instance = Mock()
        mock_smart_graph.return_value = mock_graph_instance
        
        # Mock ingestion flow with proper run method
        mock_flow_instance = Mock()
        mock_flow_instance.run.return_value = {"status": "success", "items_processed": 1}
        mock_ingestion_flow.return_value = mock_flow_instance
        
        yield {
            'get_config': mock_get_config,
            'vector_store': mock_vector_store,
            'smart_graph': mock_smart_graph,
            'graph_ops': mock_graph_ops,
            'crud': mock_crud,
            'linking': mock_linking,
            'enrichment': mock_enrichment,
            'grounding': mock_grounding,
            'personalization': mock_personalization,
            'search': mock_search,
            'monitoring': mock_monitoring,
            'evolution': mock_evolution,
            'external_resolver': mock_external_resolver,
            'ingestion_flow': mock_ingestion_flow,
            'graph_instance': mock_graph_instance
        }

@pytest.fixture
def clean_memory(mock_smartmemory_dependencies):
    """Clean SmartMemory instance for testing."""
    from smartmemory.smart_memory import SmartMemory
    memory = SmartMemory()
    
    # Configure mock behavior for common operations
    memory._graph = mock_smartmemory_dependencies['graph_instance']
    
    yield memory


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test.

    By default, cache cleanup is skipped to avoid connecting to real Redis
    during unit tests. Set SMARTMEMORY_TEST_SKIP_CACHE_CLEANUP=0 to enable.
    """
    yield
    # Cleanup logic runs after each test
    try:
        import os as _os
        if _os.environ.get("SMARTMEMORY_TEST_SKIP_CACHE_CLEANUP", "1") == "1":
            return
        # Clear any test databases, caches, etc.
        from smartmemory.utils.cache import get_cache
        cache = get_cache()
        # Provide safe fallbacks if cache doesn't expose raw redis client
        if hasattr(cache, "redis"):
            test_keys = cache.redis.keys("test_*")
            if test_keys:
                cache.redis.delete(*test_keys)
    except Exception:
        pass  # Ignore cleanup errors


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_memory_item(
        content: str = "Test content",
        memory_type: str = "semantic",
        user_id: str = "test_user",
        **kwargs
    ) -> MemoryItem:
        """Create a test memory item."""
        return MemoryItem(
            content=content,
            memory_type=memory_type,
            user_id=user_id,
            **kwargs
        )
    
    @staticmethod
    def create_memory_items(count: int = 5, **kwargs) -> List[MemoryItem]:
        """Create multiple test memory items."""
        return [
            TestDataFactory.create_memory_item(
                content=f"Test content {i}",
                **kwargs
            )
            for i in range(count)
        ]
    
    @staticmethod
    def create_graph_data() -> Dict:
        """Create test graph data."""
        return {
            "nodes": [
                {"id": "node1", "type": "CONCEPT", "properties": {"name": "Python"}},
                {"id": "node2", "type": "CONCEPT", "properties": {"name": "Programming"}},
                {"id": "node3", "type": "PERSON", "properties": {"name": "User"}}
            ],
            "edges": [
                {"source": "node1", "target": "node2", "type": "IS_A"},
                {"source": "node3", "target": "node1", "type": "KNOWS"}
            ]
        }


@pytest.fixture
def test_data_factory():
    """Test data factory instance."""
    return TestDataFactory()


# Test markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.e2e = pytest.mark.e2e
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow
