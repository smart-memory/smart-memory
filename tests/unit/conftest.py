"""
Unit test-specific fixtures and global patches to isolate from real backends.
These fixtures are ONLY imported for tests under tests/unit/.
"""
import pytest
from unittest.mock import Mock, patch


@pytest.fixture(autouse=True)
def unit_isolation_patches():
    """Autouse fixture to isolate unit tests from real services/backends.
    Patches SmartGraph, pipeline stages, MemoryIngestionFlow, and VectorStore.get.
    Also ensures configuration access returns attribute-style objects where needed.
    """
    # Patch config getter to return attribute-style object when accessed
    class AttrDict(dict):
        __getattr__ = dict.get
    
    with patch('smartmemory.configuration.get_config') as mock_get_config, \
         patch('smartmemory.smart_memory.SmartGraph') as mock_smartgraph, \
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
         patch('smartmemory.smart_memory.MemoryIngestionFlow') as mock_ingestion_flow, \
         patch('smartmemory.stores.vector.vector_store.VectorStore') as mock_vector_store:
        
        # Config returns attribute-accessible sections
        cfg = AttrDict({
            'graph_db': AttrDict({'backend_class': 'FalkorDBBackend', 'host': 'localhost', 'port': 6379}),
            'vector_store': AttrDict({'backend': 'falkordb'}),
            'cache': AttrDict({'redis': AttrDict({'host': 'localhost', 'port': 6379, 'db': 15})}),
        })
        mock_get_config.side_effect = lambda section=None: cfg.get(section) if section else cfg
        
        # Provide default mock instances
        mock_smartgraph.return_value = Mock()
        mock_graph_ops.return_value = Mock()
        mock_crud.return_value = Mock()
        mock_linking.return_value = Mock()
        mock_enrichment.return_value = Mock()
        mock_grounding.return_value = Mock()
        mock_personalization.return_value = Mock()
        mock_search.return_value = Mock()
        mock_monitoring.return_value = Mock()
        mock_evolution.return_value = Mock()
        mock_external_resolver.return_value = Mock()
        flow_instance = Mock()
        flow_instance.run.return_value = {"status": "success", "items_processed": 1}
        mock_ingestion_flow.return_value = flow_instance
        
        # VectorStore singleton get() returns a mock with basic methods
        vs = Mock()
        vs.add.return_value = Mock()
        vs.search.return_value = []
        vs.delete.return_value = Mock()
        vs.clear.return_value = Mock()
        mock_vector_store.get.return_value = vs
        
        yield
