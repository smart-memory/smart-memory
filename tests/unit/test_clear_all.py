"""Unit tests for clear_all module."""

from unittest.mock import MagicMock, patch
import argparse

import pytest

pytestmark = pytest.mark.unit

from smartmemory.clear_all import clear_graph, clear_vector_store, clear_cache, main


class TestClearGraph:
    @patch("smartmemory.graph.smartgraph.SmartGraph")
    def test_clears_graph_successfully(self, MockSmartGraph):
        mock_graph = MagicMock()
        mock_graph.clear.return_value = True
        MockSmartGraph.return_value = mock_graph

        result = clear_graph()
        assert result is True
        mock_graph.clear.assert_called_once()

    @patch("smartmemory.graph.smartgraph.SmartGraph")
    def test_returns_false_on_failure(self, MockSmartGraph):
        mock_graph = MagicMock()
        mock_graph.clear.return_value = False
        MockSmartGraph.return_value = mock_graph

        result = clear_graph()
        assert result is False

    @patch("smartmemory.graph.smartgraph.SmartGraph")
    def test_raises_on_exception(self, MockSmartGraph):
        MockSmartGraph.side_effect = RuntimeError("Connection refused")
        with pytest.raises(RuntimeError, match="Connection refused"):
            clear_graph()


class TestClearVectorStore:
    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    def test_clears_vector_store_successfully(self, MockVectorStore):
        mock_vs = MagicMock()
        mock_vs.clear.return_value = True
        MockVectorStore.return_value = mock_vs

        result = clear_vector_store()
        assert result is True
        mock_vs.clear.assert_called_once()

    @patch("smartmemory.stores.vector.vector_store.VectorStore")
    def test_raises_on_clear_failure(self, MockVectorStore):
        mock_vs = MagicMock()
        mock_vs.clear.side_effect = RuntimeError("Clear failed")
        MockVectorStore.return_value = mock_vs

        with pytest.raises(RuntimeError, match="Clear failed"):
            clear_vector_store()


class TestClearCache:
    @patch("smartmemory.utils.cache.get_cache")
    def test_clears_via_redis_client(self, mock_get_cache):
        mock_cache = MagicMock()
        mock_redis = MagicMock()
        mock_cache.redis = mock_redis
        mock_get_cache.return_value = mock_cache

        result = clear_cache()
        assert result is True
        mock_redis.flushdb.assert_called_once()

    @patch("smartmemory.utils.cache.get_cache")
    def test_no_redis_client_falls_through(self, mock_get_cache):
        mock_cache = MagicMock(spec=[])  # no .redis attribute
        mock_get_cache.return_value = mock_cache

        # No REDIS_URL env var either
        with patch.dict("os.environ", {}, clear=True):
            result = clear_cache()
            assert result is False

    @patch("smartmemory.utils.cache.get_cache")
    def test_redis_url_fallback(self, mock_get_cache):
        mock_cache = MagicMock(spec=[])  # no .redis attribute
        mock_get_cache.return_value = mock_cache

        with patch.dict("os.environ", {"REDIS_URL": "redis://localhost:6379"}):
            with patch("redis.from_url") as mock_from_url:
                mock_client = MagicMock()
                mock_from_url.return_value = mock_client

                result = clear_cache()
                assert result is True
                mock_client.flushdb.assert_called_once()

    @patch("smartmemory.utils.cache.get_cache")
    def test_raises_on_exception(self, mock_get_cache):
        mock_get_cache.side_effect = RuntimeError("Cache unavailable")
        with pytest.raises(RuntimeError, match="Cache unavailable"):
            clear_cache()


class TestMain:
    @patch("smartmemory.clear_all.clear_cache")
    @patch("smartmemory.clear_all.clear_vector_store")
    @patch("smartmemory.clear_all.clear_graph")
    def test_all_flag(self, mock_graph, mock_vector, mock_cache):
        mock_graph.return_value = True
        mock_vector.return_value = True
        mock_cache.return_value = True

        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(all=True, graph=False, vector=False, cache=False, verbose=False)):
            main()

        mock_graph.assert_called_once()
        mock_vector.assert_called_once()
        mock_cache.assert_called_once()

    @patch("smartmemory.clear_all.clear_cache")
    @patch("smartmemory.clear_all.clear_vector_store")
    @patch("smartmemory.clear_all.clear_graph")
    def test_graph_only(self, mock_graph, mock_vector, mock_cache):
        mock_graph.return_value = True

        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(all=False, graph=True, vector=False, cache=False, verbose=False)):
            main()

        mock_graph.assert_called_once()
        mock_vector.assert_not_called()
        mock_cache.assert_not_called()

    @patch("smartmemory.clear_all.clear_cache")
    @patch("smartmemory.clear_all.clear_vector_store")
    @patch("smartmemory.clear_all.clear_graph")
    def test_no_flags_defaults_to_all(self, mock_graph, mock_vector, mock_cache):
        mock_graph.return_value = True
        mock_vector.return_value = True
        mock_cache.return_value = True

        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(all=False, graph=False, vector=False, cache=False, verbose=False)):
            main()

        mock_graph.assert_called_once()
        mock_vector.assert_called_once()
        mock_cache.assert_called_once()

    @patch("smartmemory.clear_all.clear_graph")
    def test_propagates_exception(self, mock_graph):
        mock_graph.side_effect = RuntimeError("Fatal")

        with patch("argparse.ArgumentParser.parse_args",
                    return_value=argparse.Namespace(all=False, graph=True, vector=False, cache=False, verbose=False)):
            with pytest.raises(RuntimeError, match="Fatal"):
                main()
