"""
Unit tests for plugin registry.
"""

import pytest
from smartmemory.plugins.base import PluginMetadata, EnricherPlugin, EvolverPlugin
from smartmemory.plugins.registry import PluginRegistry, get_plugin_registry, reset_plugin_registry


class TestPluginRegistry:
    """Tests for PluginRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return PluginRegistry()
    
    @pytest.fixture
    def sample_enricher(self):
        """Create a sample enricher class."""
        class SampleEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="sample_enricher",
                    version="1.0.0",
                    author="Test",
                    description="Sample enricher",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {}
        
        return SampleEnricher
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample metadata."""
        return PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="Test plugin",
            plugin_type="enricher"
        )
    
    def test_register_enricher(self, registry, sample_enricher, sample_metadata):
        """Test registering an enricher."""
        registry.register_enricher("test_enricher", sample_enricher, sample_metadata)
        
        assert registry.has_plugin("test_enricher", "enricher")
        assert registry.get_enricher("test_enricher") == sample_enricher
        assert registry.get_metadata("test_enricher") == sample_metadata
    
    def test_register_duplicate_enricher_warns(self, registry, sample_enricher, sample_metadata, caplog):
        """Test that registering duplicate enricher logs warning."""
        registry.register_enricher("test", sample_enricher, sample_metadata)
        registry.register_enricher("test", sample_enricher, sample_metadata)
        
        assert "already registered" in caplog.text.lower()
    
    def test_register_enricher_with_wrong_type_raises(self, registry, sample_enricher):
        """Test that registering enricher with wrong metadata type raises error."""
        wrong_metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type="evolver"  # Wrong type
        )
        
        with pytest.raises(ValueError, match="Plugin type must be 'enricher'"):
            registry.register_enricher("test", sample_enricher, wrong_metadata)
    
    def test_list_plugins_by_type(self, registry, sample_enricher, sample_metadata):
        """Test listing plugins by type."""
        registry.register_enricher("enricher1", sample_enricher, sample_metadata)
        registry.register_enricher("enricher2", sample_enricher, sample_metadata)
        
        enrichers = registry.list_plugins("enricher")
        assert len(enrichers) == 2
        assert "enricher1" in enrichers
        assert "enricher2" in enrichers
    
    def test_list_all_plugins(self, registry, sample_enricher, sample_metadata):
        """Test listing all plugins."""
        registry.register_enricher("enricher1", sample_enricher, sample_metadata)
        
        evolver_metadata = PluginMetadata(
            name="evolver1",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type="evolver"
        )
        
        class SampleEvolver(EvolverPlugin):
            @classmethod
            def metadata(cls):
                return evolver_metadata
            
            def evolve(self, memory, logger=None):
                pass
        
        registry.register_evolver("evolver1", SampleEvolver, evolver_metadata)
        
        all_plugins = registry.list_plugins()
        assert len(all_plugins) == 2
        assert "enricher1" in all_plugins
        assert "evolver1" in all_plugins
    
    def test_has_plugin(self, registry, sample_enricher, sample_metadata):
        """Test checking if plugin exists."""
        registry.register_enricher("test", sample_enricher, sample_metadata)
        
        assert registry.has_plugin("test") is True
        assert registry.has_plugin("test", "enricher") is True
        assert registry.has_plugin("test", "evolver") is False
        assert registry.has_plugin("nonexistent") is False
    
    def test_unregister_plugin(self, registry, sample_enricher, sample_metadata):
        """Test unregistering a plugin."""
        registry.register_enricher("test", sample_enricher, sample_metadata)
        assert registry.has_plugin("test")
        
        result = registry.unregister("test")
        assert result is True
        assert not registry.has_plugin("test")
        
        # Unregistering again should return False
        result = registry.unregister("test")
        assert result is False
    
    def test_clear_all_plugins(self, registry, sample_enricher, sample_metadata):
        """Test clearing all plugins."""
        registry.register_enricher("test1", sample_enricher, sample_metadata)
        registry.register_enricher("test2", sample_enricher, sample_metadata)
        
        assert registry.count() == 2
        
        registry.clear()
        assert registry.count() == 0
    
    def test_clear_plugins_by_type(self, registry, sample_enricher, sample_metadata):
        """Test clearing plugins by type."""
        registry.register_enricher("enricher1", sample_enricher, sample_metadata)
        
        evolver_metadata = PluginMetadata(
            name="evolver1",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type="evolver"
        )
        
        class SampleEvolver(EvolverPlugin):
            @classmethod
            def metadata(cls):
                return evolver_metadata
            
            def evolve(self, memory, logger=None):
                pass
        
        registry.register_evolver("evolver1", SampleEvolver, evolver_metadata)
        
        assert registry.count() == 2
        
        registry.clear("enricher")
        assert registry.count("enricher") == 0
        assert registry.count("evolver") == 1
    
    def test_count_plugins(self, registry, sample_enricher, sample_metadata):
        """Test counting plugins."""
        assert registry.count() == 0
        
        registry.register_enricher("test1", sample_enricher, sample_metadata)
        assert registry.count() == 1
        assert registry.count("enricher") == 1
        
        registry.register_enricher("test2", sample_enricher, sample_metadata)
        assert registry.count() == 2
        assert registry.count("enricher") == 2
    
    def test_get_all_metadata(self, registry, sample_enricher, sample_metadata):
        """Test getting all metadata."""
        registry.register_enricher("test", sample_enricher, sample_metadata)
        
        all_metadata = registry.get_all_metadata()
        assert len(all_metadata) == 1
        assert "test" in all_metadata
        assert all_metadata["test"] == sample_metadata
    
    def test_registry_repr(self, registry, sample_enricher, sample_metadata):
        """Test registry string representation."""
        registry.register_enricher("test", sample_enricher, sample_metadata)
        
        repr_str = repr(registry)
        assert "PluginRegistry" in repr_str
        assert "enrichers=1" in repr_str


class TestGlobalRegistry:
    """Tests for global registry functions."""
    
    def test_get_plugin_registry_singleton(self):
        """Test that get_plugin_registry returns singleton."""
        reset_plugin_registry()  # Start fresh
        
        registry1 = get_plugin_registry()
        registry2 = get_plugin_registry()
        
        assert registry1 is registry2
    
    def test_reset_plugin_registry(self):
        """Test resetting the global registry."""
        registry1 = get_plugin_registry()
        reset_plugin_registry()
        registry2 = get_plugin_registry()
        
        assert registry1 is not registry2
