"""
Unit tests for plugin base classes and utilities.
"""

import pytest
from smartmemory.plugins.base import (
    PluginMetadata,
    PluginBase,
    EnricherPlugin,
    EvolverPlugin,
    validate_plugin_class,
    check_version_compatibility,
)


class TestPluginMetadata:
    """Tests for PluginMetadata dataclass."""
    
    def test_create_basic_metadata(self):
        """Test creating basic plugin metadata."""
        metadata = PluginMetadata(
            name="test_plugin",
            version="1.0.0",
            author="Test Author",
            description="Test plugin",
            plugin_type="enricher"
        )
        
        assert metadata.name == "test_plugin"
        assert metadata.version == "1.0.0"
        assert metadata.author == "Test Author"
        assert metadata.plugin_type == "enricher"
    
    def test_metadata_name_normalization(self):
        """Test that plugin names are normalized."""
        metadata = PluginMetadata(
            name="Test Plugin-Name",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type="enricher"
        )
        
        assert metadata.name == "test_plugin_name"
    
    def test_invalid_plugin_type(self):
        """Test that invalid plugin types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid plugin_type"):
            PluginMetadata(
                name="test",
                version="1.0.0",
                author="Test",
                description="Test",
                plugin_type="invalid_type"
            )
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            PluginMetadata(
                name="",
                version="1.0.0",
                author="Test",
                description="Test",
                plugin_type="enricher"
            )
    
    def test_metadata_with_dependencies(self):
        """Test metadata with dependencies."""
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            author="Test",
            description="Test",
            plugin_type="enricher",
            dependencies=["numpy>=1.20.0", "requests"]
        )
        
        assert len(metadata.dependencies) == 2
        assert "numpy>=1.20.0" in metadata.dependencies


class TestPluginBase:
    """Tests for PluginBase abstract class."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that PluginBase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            PluginBase()
    
    def test_valid_enricher_plugin(self):
        """Test creating a valid enricher plugin."""
        class TestEnricher(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="test_enricher",
                    version="1.0.0",
                    author="Test",
                    description="Test enricher",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {"test": "value"}
        
        enricher = TestEnricher()
        result = enricher.enrich("test item")
        assert result == {"test": "value"}
        
        metadata = TestEnricher.metadata()
        assert metadata.name == "test_enricher"
    
    def test_valid_evolver_plugin(self):
        """Test creating a valid evolver plugin."""
        class TestEvolver(EvolverPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="test_evolver",
                    version="1.0.0",
                    author="Test",
                    description="Test evolver",
                    plugin_type="evolver"
                )
            
            def evolve(self, memory, logger=None):
                pass
        
        evolver = TestEvolver()
        assert evolver.config == {}
        
        evolver_with_config = TestEvolver(config={"key": "value"})
        assert evolver_with_config.config == {"key": "value"}


class TestValidatePluginClass:
    """Tests for validate_plugin_class function."""
    
    def test_validate_valid_plugin(self):
        """Test validating a valid plugin class."""
        class ValidPlugin(EnricherPlugin):
            @classmethod
            def metadata(cls):
                return PluginMetadata(
                    name="valid",
                    version="1.0.0",
                    author="Test",
                    description="Test",
                    plugin_type="enricher"
                )
            
            def enrich(self, item, node_ids=None):
                return {}
        
        assert validate_plugin_class(ValidPlugin) is True
    
    def test_validate_abstract_base_class(self):
        """Test that abstract base classes are not valid."""
        assert validate_plugin_class(PluginBase) is False
        assert validate_plugin_class(EnricherPlugin) is False
    
    def test_validate_non_plugin_class(self):
        """Test that non-plugin classes are not valid."""
        class NotAPlugin:
            pass
        
        assert validate_plugin_class(NotAPlugin) is False
    
    def test_validate_plugin_without_metadata(self):
        """Test that plugins without metadata are not valid."""
        class NoMetadataPlugin(EnricherPlugin):
            def enrich(self, item, node_ids=None):
                return {}
        
        assert validate_plugin_class(NoMetadataPlugin) is False


class TestVersionCompatibility:
    """Tests for check_version_compatibility function."""
    
    def test_compatible_version(self):
        """Test compatible version."""
        assert check_version_compatibility("0.2.4", "0.1.0") is True
    
    def test_incompatible_min_version(self):
        """Test incompatible minimum version."""
        assert check_version_compatibility("0.1.0", "0.2.4") is False
    
    def test_compatible_with_max_version(self):
        """Test compatible with max version."""
        assert check_version_compatibility("0.2.4", "0.1.0", "0.3.0") is True
    
    def test_incompatible_max_version(self):
        """Test incompatible maximum version."""
        assert check_version_compatibility("0.4.0", "0.1.0", "0.3.0") is False
    
    def test_exact_min_version(self):
        """Test exact minimum version."""
        assert check_version_compatibility("0.1.0", "0.1.0") is True
    
    def test_exact_max_version(self):
        """Test exact maximum version."""
        assert check_version_compatibility("0.3.0", "0.1.0", "0.3.0") is True
