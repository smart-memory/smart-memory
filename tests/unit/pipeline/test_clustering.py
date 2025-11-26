import pytest
from unittest.mock import MagicMock, patch
from smartmemory.clustering.global_cluster import GlobalClustering
from smartmemory.memory.pipeline.state import ExtractionState
from smartmemory.configuration import MemoryConfig
from smartmemory.models.memory_item import MemoryItem

class TestGlobalClustering:
    @pytest.fixture
    def mock_memory(self):
        memory = MagicMock()
        memory._graph.backend.merge_nodes.return_value = True
        return memory

    @pytest.fixture
    def mock_vector_store(self):
        with patch('smartmemory.memory.pipeline.stages.clustering.VectorStore') as MockVectorStore:
            store = MockVectorStore.return_value
            # Mock backend
            backend = MagicMock()
            backend.label = "Vec_test"
            store._backend = backend
            
            # Mock graph query to return embeddings
            # Returns [id, embedding]
            backend.graph.query.return_value.result_set = [
                ["item1", [0.1, 0.1]],
                ["item2", [0.1, 0.11]], # Similar to item1
                ["item3", [0.9, 0.9]], # Different
            ]
            
            # Mock search to return similarity
            def mock_search(embedding, top_k=10, is_global=True):
                # Identify item by embedding
                current_id = None
                for row in backend.graph.query.return_value.result_set:
                    if row[1] == embedding:
                        current_id = row[0]
                        break
                
                print(f"DEBUG: Search for {current_id} (emb={embedding})")
                
                if current_id == "item1":
                    return [
                        {"id": "item1", "score": 0.0},
                        {"id": "item2", "score": 0.05}, # Very close
                        {"id": "item3", "score": 1.5}
                    ]
                elif current_id == "item2":
                    return [
                        {"id": "item2", "score": 0.0},
                        {"id": "item1", "score": 0.05},
                        {"id": "item3", "score": 1.5}
                    ]
                elif current_id == "item3":
                    return [
                        {"id": "item3", "score": 0.0},
                        {"id": "item1", "score": 1.5}
                    ]
                else:
                    # Fallback if embedding doesn't match exactly (e.g. float precision?)
                    print(f"DEBUG: Unknown embedding: {embedding}")
                    return []
            
            store.search.side_effect = mock_search
            yield store

    def test_clustering_merges_similar_items(self, mock_memory, mock_vector_store):
        stage = GlobalClustering(mock_memory)
        # Inject the mock vector store instance (since __init__ creates a new one)
        stage.vector_store = mock_vector_store
        
        # Mock memory.get to return items with names for canonical selection
        def mock_get(item_id):
            if item_id == "item1":
                return MemoryItem(item_id="item1", content="Apple", metadata={"name": "Apple"})
            elif item_id == "item2":
                return MemoryItem(item_id="item2", content="Apple Inc.", metadata={"name": "Apple Inc."}) # Longer name, should be canonical
            return None
        mock_memory.get.side_effect = mock_get

        result = stage.run()
        
        assert result["merged_count"] == 1
        assert result["clusters_found"] == 1
        
        # Verify merge_nodes called
        # Canonical should be item2 (Apple Inc.)
        mock_memory._graph.backend.merge_nodes.assert_called_once_with("item2", ["item1"])
        
        # Verify vector delete called
        mock_vector_store.delete.assert_called_once_with(["item1"])

    def test_clustering_skips_if_backend_incompatible(self, mock_memory, mock_vector_store):
        """Test that clustering skips gracefully if backend is incompatible."""
        stage = GlobalClustering(mock_memory)
        stage.vector_store = mock_vector_store
        
        # Mock incompatible backend (no label attribute)
        backend = MagicMock()
        del backend.label  # Remove label attribute
        mock_vector_store._backend = backend
        
        result = stage.run()
        
        assert result.get("skipped") == True
        assert result.get("reason") == "backend_incompatible"
