"""Unit tests for clustering.embedding — EmbeddingClusterer config and edge cases."""

from unittest.mock import patch, MagicMock

import pytest

pytestmark = pytest.mark.unit

from smartmemory.clustering.embedding import EmbeddingClusterer


class TestEmbeddingClustererInit:
    def test_defaults(self):
        c = EmbeddingClusterer()
        assert c.embedding_model_name == "all-MiniLM-L6-v2"
        assert c.cluster_size == 128
        assert c.max_clusters is None
        assert c._model is None

    def test_custom_params(self):
        c = EmbeddingClusterer(
            embedding_model="custom-model",
            cluster_size=64,
            max_clusters=10,
        )
        assert c.embedding_model_name == "custom-model"
        assert c.cluster_size == 64
        assert c.max_clusters == 10


class TestClusterItemsEdgeCases:
    def test_empty_list(self):
        c = EmbeddingClusterer()
        assert c.cluster_items([]) == []

    def test_single_item(self):
        c = EmbeddingClusterer()
        result = c.cluster_items(["only one"])
        assert result == [["only one"]]

    def test_model_import_error_raises(self):
        """_get_model raises ImportError if sentence-transformers is missing."""
        c = EmbeddingClusterer()
        with patch("smartmemory.clustering.embedding.EmbeddingClusterer._get_model",
                    side_effect=ImportError("no sentence-transformers")):
            with pytest.raises(ImportError):
                c.cluster_items(["a", "b", "c"])
