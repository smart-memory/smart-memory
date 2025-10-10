"""
Integration tests for built-in plugin discovery.

This test verifies that all built-in enrichers and evolvers can be discovered
and loaded by the PluginManager.
"""

import pytest
from smartmemory.plugins.manager import PluginManager
from smartmemory.plugins.registry import PluginRegistry


class TestBuiltinPluginDiscovery:
    """Test that built-in plugins are discovered correctly."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        registry = PluginRegistry()
        manager = PluginManager(registry=registry)
        manager.discover_plugins(
            include_builtin=True,
            include_entry_points=False,
            plugin_dirs=None
        )
        return manager
    
    def test_all_enrichers_discovered(self, manager):
        """Test that all 6 enrichers are discovered."""
        enrichers = manager.registry.list_plugins('enricher')
        
        expected_enrichers = [
            'basic_enricher',
            'sentiment_enricher',
            'temporal_enricher',
            'extract_skills_tools',
            'topic_enricher',
            'wikipedia_enricher'
        ]
        
        for enricher_name in expected_enrichers:
            assert enricher_name in enrichers, f"Enricher '{enricher_name}' not discovered"
        
        assert len(enrichers) == 6, f"Expected 6 enrichers, found {len(enrichers)}"
    
    def test_all_evolvers_discovered(self, manager):
        """Test that all 7 evolvers are discovered."""
        evolvers = manager.registry.list_plugins('evolver')
        
        expected_evolvers = [
            'working_to_episodic',
            'working_to_procedural',
            'episodic_to_semantic',
            'episodic_decay',
            'semantic_decay',
            'episodic_to_zettel',
            'zettel_prune'
        ]
        
        for evolver_name in expected_evolvers:
            assert evolver_name in evolvers, f"Evolver '{evolver_name}' not discovered"
        
        assert len(evolvers) == 7, f"Expected 7 evolvers, found {len(evolvers)}"
    
    def test_extractors_discovered(self, manager):
        """Test that extractors are discovered."""
        extractors = manager.registry.list_plugins('extractor')
        
        # Should have at least the built-in extractors
        assert len(extractors) >= 5, f"Expected at least 5 extractors, found {len(extractors)}"
        
        expected_extractors = ['spacy', 'gliner', 'rebel', 'llm', 'relik']
        for extractor_name in expected_extractors:
            assert extractor_name in extractors, f"Extractor '{extractor_name}' not discovered"
    
    def test_total_plugin_count(self, manager):
        """Test total plugin count."""
        total = len(manager.loaded_plugins)
        
        # 6 enrichers + 7 evolvers + 5 extractors + 1 grounder = 19
        assert total == 19, f"Expected 19 total plugins, found {total}"
    
    def test_enricher_metadata(self, manager):
        """Test that enricher metadata is correct."""
        # Test BasicEnricher
        meta = manager.registry.get_metadata('basic_enricher')
        assert meta is not None
        assert meta.name == 'basic_enricher'
        assert meta.version == '1.0.0'
        assert meta.plugin_type == 'enricher'
        assert meta.author == 'SmartMemory Team'
        
        # Test SentimentEnricher
        meta = manager.registry.get_metadata('sentiment_enricher')
        assert meta is not None
        assert meta.name == 'sentiment_enricher'
        assert meta.plugin_type == 'enricher'
        assert 'vaderSentiment' in meta.dependencies[0]
    
    def test_evolver_metadata(self, manager):
        """Test that evolver metadata is correct."""
        # Test WorkingToEpisodicEvolver
        meta = manager.registry.get_metadata('working_to_episodic')
        assert meta is not None
        assert meta.name == 'working_to_episodic'
        assert meta.version == '1.0.0'
        assert meta.plugin_type == 'evolver'
        assert meta.author == 'SmartMemory Team'
        
        # Test EpisodicDecayEvolver
        meta = manager.registry.get_metadata('episodic_decay')
        assert meta is not None
        assert meta.name == 'episodic_decay'
        assert meta.plugin_type == 'evolver'
    
    def test_enrichers_can_be_instantiated(self, manager):
        """Test that enrichers can be instantiated."""
        # Test BasicEnricher
        enricher_class = manager.registry.get_enricher('basic_enricher')
        assert enricher_class is not None
        enricher = enricher_class()
        assert enricher is not None
        
        # Test SentimentEnricher
        enricher_class = manager.registry.get_enricher('sentiment_enricher')
        assert enricher_class is not None
        enricher = enricher_class()
        assert enricher is not None
    
    def test_evolvers_can_be_instantiated(self, manager):
        """Test that evolvers can be instantiated."""
        from smartmemory.plugins.evolvers.episodic_decay import EpisodicDecayConfig
        from smartmemory.plugins.evolvers.working_to_episodic import WorkingToEpisodicConfig
        
        # Test EpisodicDecayEvolver
        evolver_class = manager.registry.get_evolver('episodic_decay')
        assert evolver_class is not None
        evolver = evolver_class(config=EpisodicDecayConfig())
        assert evolver is not None
        
        # Test WorkingToEpisodicEvolver
        evolver_class = manager.registry.get_evolver('working_to_episodic')
        assert evolver_class is not None
        evolver = evolver_class(config=WorkingToEpisodicConfig())
        assert evolver is not None
    
    def test_plugin_has_metadata_method(self, manager):
        """Test that plugins have metadata() classmethod."""
        # Test enricher
        enricher_class = manager.registry.get_enricher('basic_enricher')
        assert hasattr(enricher_class, 'metadata')
        assert callable(enricher_class.metadata)
        
        # Test evolver
        evolver_class = manager.registry.get_evolver('episodic_decay')
        assert hasattr(evolver_class, 'metadata')
        assert callable(evolver_class.metadata)
    
    def test_all_plugins_have_valid_metadata(self, manager):
        """Test that all plugins have valid metadata."""
        all_plugins = manager.registry.list_plugins()
        
        for plugin_name in all_plugins:
            meta = manager.registry.get_metadata(plugin_name)
            assert meta is not None, f"Plugin '{plugin_name}' has no metadata"
            assert meta.name == plugin_name, f"Plugin name mismatch for '{plugin_name}'"
            assert meta.version, f"Plugin '{plugin_name}' has no version"
            assert meta.plugin_type in ['enricher', 'extractor', 'evolver', 'embedding', 'grounder'], \
                f"Plugin '{plugin_name}' has invalid type: {meta.plugin_type}"


class TestBackwardCompatibility:
    """Test that old import paths and usage patterns still work."""
    
    def test_old_enricher_imports_work(self):
        """Test that old enricher imports still work."""
        from smartmemory.plugins.enrichers import BasicEnricher, SentimentEnricher
        
        # Should be able to instantiate
        basic = BasicEnricher()
        sentiment = SentimentEnricher()
        
        assert basic is not None
        assert sentiment is not None
    
    def test_old_evolver_imports_work(self):
        """Test that old evolver imports still work."""
        from smartmemory.plugins.evolvers.episodic_decay import (
            EpisodicDecayEvolver,
            EpisodicDecayConfig
        )
        from smartmemory.plugins.evolvers.working_to_episodic import (
            WorkingToEpisodicEvolver,
            WorkingToEpisodicConfig
        )
        
        # Should be able to instantiate
        decay = EpisodicDecayEvolver(config=EpisodicDecayConfig())
        working = WorkingToEpisodicEvolver(config=WorkingToEpisodicConfig())
        
        assert decay is not None
        assert working is not None
    


class TestPluginFunctionality:
    """Test that plugins actually work correctly."""
    
    def test_enricher_enrich_method_works(self):
        """Test that enricher enrich() method works."""
        from smartmemory.plugins.enrichers import BasicEnricher
        
        enricher = BasicEnricher()
        
        # Mock item
        class MockItem:
            content = "This is a test. It has multiple sentences."
        
        result = enricher.enrich(MockItem())
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'summary' in result
    
    def test_enricher_with_config_works(self):
        """Test that enrichers work with config."""
        from smartmemory.plugins.enrichers.basic import BasicEnricher, BasicEnricherConfig
        
        config = BasicEnricherConfig(enable_summary=True, enable_entity_tags=False)
        enricher = BasicEnricher(config=config)
        
        class MockItem:
            content = "Test content."
        
        result = enricher.enrich(MockItem())
        assert 'summary' in result
