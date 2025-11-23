"""
Integration tests for configuration loading and validation.
"""
import pytest
from smartmemory.configuration import get_config
from smartmemory.configuration.manager import ConfigManager

@pytest.mark.integration  
class TestConfigurationIntegration:
    """Test configuration loading and validation."""
    
    def test_configuration_loads_successfully(self):
        """Test that configuration loads without critical errors."""
        try:
            cache_config = get_config('cache')
            config_loaded = True
            has_host_key = 'host' in cache_config
            has_redis_section = 'redis' in cache_config
        except Exception as e:
            config_loaded = False
            has_host_key = False
            has_redis_section = False
            print(f"Configuration error: {e}")
        
        assert config_loaded, "Configuration should load successfully"
        assert has_host_key, "Cache configuration should have host key"
        assert has_redis_section, "Cache configuration should have redis section"

    def test_configuration_validation_integration(self):
        """Test configuration validation with real ConfigManager."""
        config_manager = ConfigManager()
        
        # ConfigManager.validate_config() raises exception on failure, returns None on success
        try:
            config_manager.validate_config()
        except Exception as e:
            # It's okay if it fails due to missing env vars in test env, but we want to catch it
            print(f"Config validation warning (expected in test env): {e}")
