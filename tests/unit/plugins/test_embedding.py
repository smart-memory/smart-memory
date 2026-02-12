"""Unit tests for EmbeddingService token tracking (CFS-1a)."""

import threading
import pytest

pytestmark = pytest.mark.unit


class TestEmbeddingUsageTracking:
    """Tests for embedding token usage tracking."""

    def test_get_last_embedding_usage_consume_once(self):
        """get_last_embedding_usage returns usage once, then None."""
        from smartmemory.plugins.embedding import (
            EmbeddingUsage,
            _set_last_usage,
            get_last_embedding_usage,
        )

        # Set usage
        _set_last_usage(EmbeddingUsage(prompt_tokens=100, total_tokens=100, model="ada", cached=False))

        # First call returns usage
        usage = get_last_embedding_usage()
        assert usage is not None
        assert usage["prompt_tokens"] == 100
        assert usage["total_tokens"] == 100
        assert usage["model"] == "ada"
        assert usage["cached"] is False

        # Second call returns None (consumed)
        usage2 = get_last_embedding_usage()
        assert usage2 is None

    def test_get_last_embedding_usage_cached_flag(self):
        """Cached embedding calls are tracked with cached=True."""
        from smartmemory.plugins.embedding import (
            EmbeddingUsage,
            _set_last_usage,
            get_last_embedding_usage,
        )

        _set_last_usage(EmbeddingUsage(prompt_tokens=0, total_tokens=0, model="ada", cached=True))

        usage = get_last_embedding_usage()
        assert usage["cached"] is True
        assert usage["prompt_tokens"] == 0

    def test_get_last_embedding_usage_none_when_not_set(self):
        """get_last_embedding_usage returns None when no usage has been recorded."""
        from smartmemory.plugins.embedding import get_last_embedding_usage

        # Clear any existing usage by consuming it
        get_last_embedding_usage()

        # Now should be None
        usage = get_last_embedding_usage()
        assert usage is None

    def test_embedding_usage_thread_isolation(self):
        """Usage is isolated per thread."""
        from smartmemory.plugins.embedding import (
            EmbeddingUsage,
            _set_last_usage,
            get_last_embedding_usage,
        )

        results = {}

        def thread_a():
            _set_last_usage(EmbeddingUsage(prompt_tokens=100, model="thread-a", cached=False))
            results["a"] = get_last_embedding_usage()

        def thread_b():
            _set_last_usage(EmbeddingUsage(prompt_tokens=200, model="thread-b", cached=True))
            results["b"] = get_last_embedding_usage()

        t1 = threading.Thread(target=thread_a)
        t2 = threading.Thread(target=thread_b)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each thread should see its own usage
        assert results["a"]["prompt_tokens"] == 100
        assert results["a"]["model"] == "thread-a"
        assert results["b"]["prompt_tokens"] == 200
        assert results["b"]["model"] == "thread-b"


class TestEmbeddingUsageDataclass:
    """Tests for EmbeddingUsage dataclass."""

    def test_defaults(self):
        """EmbeddingUsage has sensible defaults."""
        from smartmemory.plugins.embedding import EmbeddingUsage

        usage = EmbeddingUsage()
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0
        assert usage.model == ""
        assert usage.cached is False

    def test_custom_values(self):
        """EmbeddingUsage stores custom values."""
        from smartmemory.plugins.embedding import EmbeddingUsage

        usage = EmbeddingUsage(
            prompt_tokens=500,
            total_tokens=500,
            model="text-embedding-3-large",
            cached=True,
        )
        assert usage.prompt_tokens == 500
        assert usage.model == "text-embedding-3-large"
        assert usage.cached is True
