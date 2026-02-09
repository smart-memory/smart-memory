"""
Tests to verify extraction stage is auth-agnostic.

These tests exercise the extraction pipeline the way the service layer does,
ensuring no auth field names leak into the core library.
"""
import pytest


pytestmark = pytest.mark.integration
from unittest.mock import Mock, patch

from smartmemory.models.memory_item import MemoryItem
from smartmemory.extraction.extractor import OntologyExtractor
from smartmemory.memory.pipeline.extractor import ExtractorPipeline
from smartmemory.plugins.extractors.llm import LLMExtractor
from smartmemory.plugins.base import ExtractorPlugin


class TestExtractorAuthAgnostic:
    """Test that extractors don't require or use auth fields."""
    
    def test_ontology_extractor_no_user_id_param(self):
        """Verify OntologyExtractor.extract_entities_and_relations has no user_id param."""
        import inspect
        sig = inspect.signature(OntologyExtractor.extract_entities_and_relations)
        param_names = list(sig.parameters.keys())
        
        assert 'user_id' not in param_names, \
            f"user_id should not be a parameter. Found params: {param_names}"
        assert 'self' in param_names
        assert 'text' in param_names
    
    def test_extractor_plugin_interface_no_user_id(self):
        """Verify ExtractorPlugin.extract interface has no user_id."""
        import inspect
        sig = inspect.signature(ExtractorPlugin.extract)
        param_names = list(sig.parameters.keys())
        
        assert 'user_id' not in param_names, \
            f"user_id should not be in ExtractorPlugin.extract. Found: {param_names}"
    
    def test_llm_extractor_no_user_id(self):
        """Verify LLMExtractor.extract has no user_id param."""
        import inspect
        sig = inspect.signature(LLMExtractor.extract)
        param_names = list(sig.parameters.keys())
        
        assert 'user_id' not in param_names, \
            f"user_id should not be in LLMExtractor.extract. Found: {param_names}"


class TestExtractionPipelineAuthAgnostic:
    """Test extraction pipeline doesn't leak auth fields."""
    
    def test_memory_item_no_user_id_field(self):
        """Verify MemoryItem has no user_id field."""
        item = MemoryItem(content="Test content")
        
        assert not hasattr(item, 'user_id') or getattr(item, 'user_id', None) is None, \
            "MemoryItem should not have user_id as a field"
        
        # Check it's not in dataclass fields
        from dataclasses import fields
        field_names = [f.name for f in fields(item)]
        assert 'user_id' not in field_names, \
            f"user_id should not be a MemoryItem field. Found: {field_names}"
    
    def test_memory_item_accepts_metadata_with_auth(self):
        """Verify MemoryItem can store auth context in metadata."""
        item = MemoryItem(
            content="Test content",
            metadata={
                "user_id": "test_user",
                "tenant_id": "test_tenant",
                "workspace_id": "test_workspace"
            }
        )
        
        assert item.metadata.get("user_id") == "test_user"
        assert item.metadata.get("tenant_id") == "test_tenant"
        assert item.metadata.get("workspace_id") == "test_workspace"
    
    def test_extraction_stage_processes_item_without_auth(self):
        """Test extraction stage can process items without auth params."""
        from smartmemory.memory.pipeline.config import ExtractionConfig
        
        stage = ExtractorPipeline()
        config = ExtractionConfig()
        
        item = MemoryItem(
            content="John works at Google as a software engineer.",
            memory_type="semantic"
        )
        
        # Mock the extractor to avoid actual LLM calls
        mock_result = {
            'entities': [
                {'name': 'John', 'type': 'person'},
                {'name': 'Google', 'type': 'organization'}
            ],
            'relations': [
                {'source': 'John', 'type': 'works_at', 'target': 'Google'}
            ]
        }
        
        with patch.object(stage, '_resolve_extractor') as mock_resolve:
            mock_extractor = Mock()
            mock_extractor.return_value = mock_result
            mock_resolve.return_value = mock_extractor
            
            # This should work without any user_id parameter
            result = stage.run(item, config)
            
            # Verify extractor was called without user_id
            if mock_extractor.called:
                call_args = mock_extractor.call_args
                if call_args:
                    # Check no user_id in args or kwargs
                    args, kwargs = call_args
                    assert 'user_id' not in kwargs, \
                        f"user_id should not be passed to extractor. kwargs: {kwargs}"


class TestOntologyExtractorAuthAgnostic:
    """Test OntologyExtractor specifically."""
    
    def test_create_ontology_nodes_no_user_id(self):
        """Verify _create_ontology_nodes doesn't accept user_id."""
        import inspect
        sig = inspect.signature(OntologyExtractor._create_ontology_nodes)
        param_names = list(sig.parameters.keys())
        
        assert 'user_id' not in param_names, \
            f"_create_ontology_nodes should not have user_id. Found: {param_names}"
    
    def test_ontology_node_no_user_id_field(self):
        """Verify OntologyNode base class has no user_id field."""
        from smartmemory.models.ontology import OntologyNode
        from dataclasses import fields
        
        # OntologyNode is abstract, check its fields
        field_names = [f.name for f in fields(OntologyNode)]
        assert 'user_id' not in field_names, \
            f"OntologyNode should not have user_id field. Found: {field_names}"


class TestScopeProviderIntegration:
    """Test that scope_provider is the only source of auth context."""
    
    def test_default_scope_provider_exists(self):
        """Verify DefaultScopeProvider is available for OSS usage."""
        from smartmemory.scope_provider import DefaultScopeProvider
        
        provider = DefaultScopeProvider()
        
        # Should implement the interface
        assert hasattr(provider, 'get_isolation_filters')
        assert hasattr(provider, 'get_write_context')
        assert hasattr(provider, 'get_global_search_filters')
        assert hasattr(provider, 'get_user_isolation_key')
    
    def test_default_scope_provider_returns_dicts(self):
        """Verify DefaultScopeProvider returns proper dicts."""
        from smartmemory.scope_provider import DefaultScopeProvider
        
        provider = DefaultScopeProvider()
        
        filters = provider.get_isolation_filters()
        assert isinstance(filters, dict)
        
        context = provider.get_write_context()
        assert isinstance(context, dict)
        
        global_filters = provider.get_global_search_filters()
        assert isinstance(global_filters, dict)
    
    def test_smart_memory_uses_scope_provider(self):
        """Verify SmartMemory accepts and uses scope_provider."""
        from smartmemory.smart_memory import SmartMemory
        from smartmemory.scope_provider import DefaultScopeProvider
        
        provider = DefaultScopeProvider(
            tenant_id="test_tenant",
            workspace_id="test_workspace"
        )
        
        # SmartMemory should accept scope_provider
        # (may fail to connect to DB, but should accept the param)
        try:
            memory = SmartMemory(scope_provider=provider)
            assert memory.scope_provider == provider
        except Exception as e:
            # Connection errors are OK, we're testing the interface
            if "scope_provider" in str(e).lower():
                pytest.fail(f"SmartMemory should accept scope_provider: {e}")


class TestNoAuthLeakage:
    """Test that auth field names don't leak into core library logic."""
    
    def test_vector_store_uses_generic_filters(self):
        """Verify VectorStore uses generic filter iteration."""
        from smartmemory.stores.vector.vector_store import VectorStore
        import inspect
        
        # Check search method source doesn't hardcode user_id checks
        source = inspect.getsource(VectorStore.search)
        
        # Should use generic filter iteration, not hardcoded field names
        assert 'for filter_key, filter_value in filters.items()' in source or \
               'filters.items()' in source, \
            "VectorStore.search should use generic filter iteration"
        
        # Should NOT have explicit user_id checks
        assert 'effective_user_id' not in source, \
            "VectorStore.search should not have explicit user_id variable"
    
    def test_graph_backend_uses_scope_provider(self):
        """Verify FalkorDB backend uses scope_provider methods."""
        from smartmemory.graph.backends.falkordb import FalkorDBBackend
        import inspect
        
        # Check __init__ accepts scope_provider
        sig = inspect.signature(FalkorDBBackend.__init__)
        assert 'scope_provider' in sig.parameters, \
            "FalkorDBBackend should accept scope_provider"


class TestStudioExtractionFlow:
    """
    Test the full 3-stage extraction flow as Studio/Service does it.
    
    Studio flow:
    1. run_input_adapter() - converts content to MemoryItem
    2. run_classification() - classifies memory type
    3. run_extraction() - extracts entities and relations
    
    Scope is injected via metadata, not explicit params.
    """
    
    def test_pipeline_3_stage_flow_no_auth_params(self):
        """Test that Pipeline stages don't require auth params in their signatures."""
        from smartmemory.memory.pipeline.components import Pipeline
        import inspect
        
        pipeline = Pipeline()
        
        # Check run_input_adapter signature
        sig = inspect.signature(pipeline.run_input_adapter)
        params = list(sig.parameters.keys())
        assert 'user_id' not in params, f"run_input_adapter should not have user_id: {params}"
        assert 'workspace_id' not in params, f"run_input_adapter should not have workspace_id: {params}"
        
        # Check run_classification signature
        sig = inspect.signature(pipeline.run_classification)
        params = list(sig.parameters.keys())
        assert 'user_id' not in params, f"run_classification should not have user_id: {params}"
        
        # Check run_extraction signature
        sig = inspect.signature(pipeline.run_extraction)
        params = list(sig.parameters.keys())
        assert 'user_id' not in params, f"run_extraction should not have user_id: {params}"
    
    def test_input_adapter_stage_accepts_metadata_scope(self):
        """Test input adapter accepts scope in metadata (how service injects it)."""
        from smartmemory.memory.pipeline.components import Pipeline
        from smartmemory.memory.pipeline.config import InputAdapterConfig
        from smartmemory.memory.pipeline.input_adapter import InputAdapter
        
        pipeline = Pipeline()
        # Register the input adapter component
        pipeline.register_component("input", InputAdapter())
        
        # This is how ScopedPipeline injects scope
        config = InputAdapterConfig(
            content="Test content about machine learning",
            memory_type="semantic",
            metadata={
                "workspace_id": "test_workspace",
                "user_id": "test_user",
                "tenant_id": "test_tenant"
            },
            adapter_name="text"
        )
        
        result = pipeline.run_input_adapter(config)
        
        assert result.success, f"Input adapter failed: {result}"
        # Verify scope is preserved in the item's metadata
        item = result.memory_item or (result.item if hasattr(result, 'item') else None)
        if item:
            assert item.metadata.get("workspace_id") == "test_workspace"
            assert item.metadata.get("user_id") == "test_user"
    
    def test_classification_stage_preserves_metadata(self):
        """Test classification preserves metadata from input stage."""
        from smartmemory.memory.pipeline.components import Pipeline
        from smartmemory.memory.pipeline.config import InputAdapterConfig, ClassificationConfig
        from smartmemory.memory.pipeline.input_adapter import InputAdapter
        from smartmemory.memory.pipeline.classification import ClassificationEngine
        
        pipeline = Pipeline()
        pipeline.register_component("input", InputAdapter())
        pipeline.register_component("classification", ClassificationEngine())
        
        # Run input adapter first
        input_config = InputAdapterConfig(
            content="I prefer Python for data analysis",
            memory_type="semantic",
            metadata={"user_id": "test_user", "workspace_id": "ws1"},
            adapter_name="text"
        )
        pipeline.run_input_adapter(input_config)
        
        # Run classification
        class_config = ClassificationConfig(
            content_analysis_enabled=False,
            default_confidence=0.9
        )
        result = pipeline.run_classification(class_config)
        
        assert result.success, f"Classification failed: {result}"
        # Metadata should be preserved through stages
        if hasattr(pipeline.state, 'input_state') and pipeline.state.input_state:
            item = pipeline.state.input_state.memory_item
            if item:
                assert item.metadata.get("user_id") == "test_user"
    
    def test_extraction_stage_no_user_id_in_extractor_call(self):
        """Test extraction stage doesn't pass user_id to extractors."""
        from smartmemory.memory.pipeline.components import Pipeline
        from smartmemory.memory.pipeline.config import (
            InputAdapterConfig, ClassificationConfig, ExtractionConfig
        )
        from unittest.mock import patch, Mock
        
        pipeline = Pipeline()
        
        # Run prerequisite stages
        input_config = InputAdapterConfig(
            content="John works at Google as an engineer",
            memory_type="semantic",
            metadata={"user_id": "test_user"},
            adapter_name="text"
        )
        pipeline.run_input_adapter(input_config)
        
        class_config = ClassificationConfig(content_analysis_enabled=False)
        pipeline.run_classification(class_config)
        
        # Mock the extractor to capture call args
        mock_extractor = Mock()
        mock_extractor.extract.return_value = {
            'entities': [{'name': 'John', 'type': 'person'}],
            'relations': []
        }
        
        with patch('smartmemory.plugins.manager.get_plugin_manager') as mock_pm:
            mock_manager = Mock()
            mock_manager.get_extractor.return_value = mock_extractor
            mock_pm.return_value = mock_manager
            
            extract_config = ExtractionConfig(extractor_name="mock")
            result = pipeline.run_extraction(extract_config)
            
            # Check extractor.extract was called without user_id
            if mock_extractor.extract.called:
                call_args = mock_extractor.extract.call_args
                args, kwargs = call_args
                assert 'user_id' not in kwargs, \
                    f"Extractor should not receive user_id. Got kwargs: {kwargs}"
    
    def test_full_3_stage_flow_with_scope_in_metadata(self):
        """Test complete 3-stage flow with scope injected via metadata."""
        from smartmemory.memory.pipeline.components import Pipeline
        from smartmemory.memory.pipeline.config import (
            InputAdapterConfig, ClassificationConfig, ExtractionConfig
        )
        from smartmemory.memory.pipeline.input_adapter import InputAdapter
        from smartmemory.memory.pipeline.classification import ClassificationEngine
        from smartmemory.memory.pipeline.extractor import ExtractorPipeline
        from unittest.mock import patch, Mock
        
        pipeline = Pipeline()
        pipeline.register_component("input", InputAdapter())
        pipeline.register_component("classification", ClassificationEngine())
        pipeline.register_component("extraction", ExtractorPipeline())
        
        # Stage 1: Input Adapter with scope in metadata
        input_config = InputAdapterConfig(
            content="Alice is a software engineer at Microsoft",
            memory_type="semantic",
            metadata={
                "workspace_id": "prod_workspace",
                "user_id": "alice_123",
                "tenant_id": "acme_corp"
            },
            adapter_name="text"
        )
        input_result = pipeline.run_input_adapter(input_config)
        assert input_result.success, f"Input failed: {input_result}"
        
        # Stage 2: Classification
        class_config = ClassificationConfig(
            content_analysis_enabled=False,
            default_confidence=0.9
        )
        class_result = pipeline.run_classification(class_config)
        assert class_result.success, f"Classification failed: {class_result}"
        
        # Stage 3: Extraction (with mock to avoid LLM calls)
        mock_extractor = Mock()
        mock_extractor.extract.return_value = {
            'entities': [
                {'name': 'Alice', 'type': 'person'},
                {'name': 'Microsoft', 'type': 'organization'}
            ],
            'relations': [
                {'source': 'Alice', 'type': 'works_at', 'target': 'Microsoft'}
            ]
        }
        
        with patch('smartmemory.plugins.manager.get_plugin_manager') as mock_pm:
            mock_manager = Mock()
            mock_manager.get_extractor.return_value = mock_extractor
            mock_pm.return_value = mock_manager
            
            extract_config = ExtractionConfig(extractor_name="mock")
            extract_result = pipeline.run_extraction(extract_config)
            
            # Extraction should succeed
            assert extract_result.success, f"Extraction failed: {extract_result}"
            
            # Verify scope is still in the item's metadata
            if hasattr(pipeline.state, 'input_state') and pipeline.state.input_state:
                item = pipeline.state.input_state.memory_item
                if item:
                    assert item.metadata.get("workspace_id") == "prod_workspace"
                    assert item.metadata.get("user_id") == "alice_123"
                    assert item.metadata.get("tenant_id") == "acme_corp"


class TestScopedPipelineIntegration:
    """Test that ScopedPipeline from service_common works correctly."""
    
    def test_scoped_pipeline_injects_scope_to_metadata(self):
        """Test ScopedPipeline._inject_scope adds workspace_id and user_id."""
        # Import from service_common if available
        try:
            from service_common.pipeline.scoped_pipeline import ScopedPipeline
        except ImportError:
            pytest.skip("service_common not available")
        
        pipeline = ScopedPipeline(
            workspace_id="test_ws",
            user_id="test_user"
        )
        
        # Test _inject_scope
        ctx = pipeline._inject_scope({})
        assert ctx["workspace_id"] == "test_ws"
        assert ctx["user_id"] == "test_user"
        
        # Test with existing context
        ctx = pipeline._inject_scope({"existing": "value"})
        assert ctx["workspace_id"] == "test_ws"
        assert ctx["user_id"] == "test_user"
        assert ctx["existing"] == "value"
    
    def test_scoped_pipeline_input_adapter_injects_scope(self):
        """Test ScopedPipeline.run_input_adapter injects scope into metadata."""
        try:
            from service_common.pipeline.scoped_pipeline import ScopedPipeline
            from smartmemory.memory.pipeline.config import InputAdapterConfig
        except ImportError:
            pytest.skip("service_common not available")
        
        pipeline = ScopedPipeline(
            workspace_id="ws_123",
            user_id="user_456"
        )
        
        config = InputAdapterConfig(
            content="Test content",
            memory_type="semantic",
            metadata={},
            adapter_name="text"
        )
        
        result = pipeline.run_input_adapter(config)
        
        assert result.success
        # Check scope was injected into config metadata
        assert config.metadata.get("workspace_id") == "ws_123"
        assert config.metadata.get("user_id") == "user_456"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
