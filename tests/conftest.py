"""
Test configuration and fixtures for SmartMemory test suite.
Provides common fixtures, test data, and setup/teardown functionality.
"""

import pytest
import tempfile
import shutil
import os


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "harness: marks tests as error harness tests")
    config.addinivalue_line("markers", "contract: marks tests as contract tests (API stability)")
    config.addinivalue_line("markers", "invariant: marks tests as invariant tests (logic kernels)")
    config.addinivalue_line("markers", "golden: marks tests as golden flow tests")
    config.addinivalue_line("markers", "slow: marks tests as slow (require external services)")


@pytest.fixture(scope="session")
def test_config():
    """Test configuration with isolated backends for unit tests."""
    return {
        "graph_db": {
            "backend_class": "FalkorDBBackend",
            "host": "localhost",
            "port": 9010,
            "database": "test_smartmemory",
        },
        "vector_store": {"backend": "falkordb", "collection_name": "test_collection"},
        "cache": {
            "redis": {
                "host": "localhost",
                "port": 9012,
                "db": 15,  # Use separate test database
            }
        },
    }


@pytest.fixture(scope="session")
def integration_config():
    """Real configuration for integration tests - no mocks."""
    return {
        "graph_db": {
            "backend_class": "FalkorDBBackend",
            "host": "localhost",
            "port": 9010,
            "database": "integration_test_smartmemory",
        },
        "vector_store": {"backend": "falkordb", "collection_name": "integration_test_collection"},
        "cache": {
            "redis": {
                "host": "localhost",
                "port": 9012,
                "db": 14,  # Separate integration test database
            }
        },
        "extractors": {"openai": {"api_key": os.getenv("OPENAI_API_KEY", "test-key"), "model": "gpt-3.5-turbo"}},
        "embedding": {"provider": "openai", "model": "text-embedding-ada-002"},
    }


@pytest.fixture
def real_smartmemory_for_integration(integration_config):
    """Real SmartMemory instance for integration tests - NO MOCKS."""
    # Import here to avoid circular imports
    from smartmemory.smart_memory import SmartMemory

    # Set environment variables to match integration config/docker services
    os.environ["FALKORDB_PORT"] = str(integration_config["graph_db"]["port"])
    os.environ["REDIS_PORT"] = str(integration_config["cache"]["redis"]["port"])
    os.environ["VECTOR_BACKEND"] = "falkordb"  # Use FalkorDB as default if Chroma is missing

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
    if "SMARTMEMORY_ENV" in os.environ:
        del os.environ["SMARTMEMORY_ENV"]

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
