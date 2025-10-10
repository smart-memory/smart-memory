"""
Unit tests for plugin manager.
"""

import pytest
from smartmemory.plugins.base import PluginMetadata, EnricherPlugin
from smartmemory.plugins.manager import PluginManager
from smartmemory.plugins.registry import PluginRegistry


class TestPluginManager:
    """Tests for PluginManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test."""
        registry = PluginRegistry()
        return PluginManager(registry=registry)
    
    def test_manager_initialization(self, manager):
        """Test that manager initializes correctly."""
        assert manager.registry is not None
        assert len(manager.loaded_plugins) == 0
    
    def test_discover_plugins_no_sources(self, manager):
        """Test discovery with all sources disabled."""
        manager.discover_plugins(
            include_builtin=False,
            include_entry_points=False,
            plugin_dirs=None
        )
        
        # Should have no plugins loaded
        assert len(manager.loaded_plugins) == 0
    
    def test_discover_builtin_plugins(self, manager):
        """Test discovering built-in plugins."""
        # Note: This will try to load actual built-in plugins
        # For now, we expect it to handle missing plugins gracefully
        manager.discover_plugins(
            include_builtin=True,
            include_entry_points=False,
            plugin_dirs=None
        )
        
        # Should complete without errors
        # The actual count depends on whether built-in plugins have metadata() yet
        assert isinstance(manager.loaded_plugins, set)
    
    def test_unload_plugin(self, manager):
        """Test unloading a plugin."""
        # Create and register a test plugin
        class TestEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="test_enricher",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {}
        
        metadata = TestEnricher.metadata()
        manager.registry.register_enricher("test_enricher", TestEnricher, metadata)
        manager._loaded_plugins.add("test_enricher")
        
        assert "test_enricher" in manager.loaded_plugins
        
        # Unload it
        result = manager.unload_plugin("test_enricher")
        assert result is True
        assert "test_enricher" not in manager.loaded_plugins
        
        # Try unloading again
        result = manager.unload_plugin("test_enricher")
        assert result is False
    
    def test_loaded_plugins_is_copy(self, manager):
        """Test that loaded_plugins returns a copy."""
        plugins1 = manager.loaded_plugins
        plugins2 = manager.loaded_plugins
        
        assert plugins1 is not plugins2
        assert plugins1 == plugins2


class TestPluginManagerIntegration:
    """Integration tests for plugin manager."""
    
    def test_manager_with_registry_integration(self):
        """Test that manager properly integrates with registry."""
        registry = PluginRegistry()
        manager = PluginManager(registry=registry)
        
        # Create a test plugin
        class TestEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="integration_test",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {}
        
        # Manually register (simulating what discover would do)
        metadata = TestEnricher.metadata()
        manager.registry.register_enricher("integration_test", TestEnricher, metadata)
        manager._loaded_plugins.add("integration_test")
        
        # Verify it's accessible through registry
        assert registry.has_plugin("integration_test")
        assert registry.get_enricher("integration_test") == TestEnricher
        
        # Verify manager tracks it
        assert "integration_test" in manager.loaded_plugins
