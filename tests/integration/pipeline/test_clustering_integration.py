import pytest
from unittest.mock import MagicMock
from smartmemory.models.memory_item import MemoryItem

@pytest.mark.integration
class TestClusteringIntegration:
    """Integration tests for GlobalClustering stage within SmartMemory."""

    def test_run_clustering_integration(self, clean_memory, mock_smartmemory_dependencies):
        """Test run_clustering method on SmartMemory instance."""
        memory = clean_memory
        
        # Create a mock vector store instance
        mock_vector_store = MagicMock()
        
        # Mock backend for GlobalClustering
        backend = MagicMock()
        backend.label = "Vec_test"
        mock_vector_store._backend = backend
        
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
                return []
        
        mock_vector_store.search.side_effect = mock_search
        mock_vector_store.delete.return_value = None
        
        # Inject mock vector store into clustering stage
        memory._clustering.vector_store = mock_vector_store
        
        # Mock memory.get directly
        def mock_get(item_id):
            if item_id == "item1":
                return MemoryItem(item_id="item1", content="Apple", metadata={"name": "Apple"})
            elif item_id == "item2":
                return MemoryItem(item_id="item2", content="Apple Inc.", metadata={"name": "Apple Inc."})
            return None
        memory.get = mock_get
        
        # Mock merge_nodes on backend
        mock_graph = mock_smartmemory_dependencies['graph_instance']
        mock_graph.backend.merge_nodes.return_value = True

        # Run clustering
        result = memory.run_clustering()
        
        # Verify result
        assert result["merged_count"] == 1
        assert result["clusters_found"] == 1
        
        # Verify merge_nodes called
        mock_graph.backend.merge_nodes.assert_called_once_with("item2", ["item1"])
        
        # Verify vector delete called
        mock_vector_store.delete.assert_called_once_with(["item1"])
