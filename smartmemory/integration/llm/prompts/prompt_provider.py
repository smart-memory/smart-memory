"""
Abstract prompt provider interface for dependency injection.

This allows smartmemory to remain agnostic about prompt sources while
enabling different services to inject their own prompt providers:
- Smart-memory service: Config-based prompts from prompts.json
- Smart-memory-studio service: MongoDB-based prompts with user/workspace overrides
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PromptProvider(ABC):
    """Abstract interface for prompt providers with dependency injection."""

    @abstractmethod
    def get_prompt_value(self, path: str, default: Any = None) -> Any:
        """Retrieve a prompt value by dot-notation path.
        
        Args:
            path: Dot-notation path like 'extractor.ontology.system_template'
            default: Default value if path not found
            
        Returns:
            The prompt value or default
        """
        pass

    @abstractmethod
    def apply_placeholders(self, template: str, mapping: Dict[str, str]) -> str:
        """Replace placeholder keys like <KEY> or {{KEY}} with their values.
        
        Args:
            template: Template string with placeholders
            mapping: Dictionary of key-value replacements
            
        Returns:
            Template with placeholders replaced
        """
        pass

    @abstractmethod
    def reload_prompts(self) -> None:
        """Reload prompts from source (config file, database, etc.)."""
        pass

    @abstractmethod
    def list_available_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts with metadata.
        
        Returns:
            List of prompt dictionaries with keys: path, name, description, category
        """
        pass


class ConfigPromptProvider(PromptProvider):
    """Config-based prompt provider using prompts.json file."""

    def __init__(self):
        # Import here to avoid circular imports
        from smartmemory.integration.llm.prompts.prompts_loader import get_prompt_value as _get_prompt_value, apply_placeholders as _apply_placeholders, get_prompts_config
        self._get_prompt_value = _get_prompt_value
        self._apply_placeholders = _apply_placeholders
        self._get_prompts_config = get_prompts_config

    def get_prompt_value(self, path: str, default: Any = None) -> Any:
        """Get prompt value from config file."""
        return self._get_prompt_value(path, default)

    def apply_placeholders(self, template: str, mapping: Dict[str, str]) -> str:
        """Apply placeholders using config-based logic."""
        return self._apply_placeholders(template, mapping)

    def reload_prompts(self) -> None:
        """Reload prompts from config file."""
        config = self._get_prompts_config()
        config.reload_if_stale(force=True)

    def list_available_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts from config file."""
        try:
            config = self._get_prompts_config()
            prompts_data = config.prompts

            available_prompts = []
            self._extract_prompts_recursive(prompts_data, "", available_prompts)

            return available_prompts

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error listing available prompts: {e}")
            return []

    def _extract_prompts_recursive(self, data: Dict[str, Any], path_prefix: str, results: List[Dict[str, Any]]) -> None:
        """Recursively extract prompt paths from config data."""
        for key, value in data.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key

            if isinstance(value, dict):
                # Check if this dict contains template fields
                template_fields = ["system_template", "user_template", "assistant_template",
                                   "json_fallback_template", "prompt_template", "analysis_template",
                                   "improvement_template", "domain_enrichment_template",
                                   "validation_template", "concept_template", "relation_template"]

                has_templates = any(field in value for field in template_fields)

                if has_templates:
                    # This is a prompt definition
                    category = self._get_category_from_path(current_path)
                    results.append({
                        "path": current_path,
                        "name": current_path.replace(".", " ").title(),
                        "description": f"Config prompt for {current_path}",
                        "category": category
                    })

                    # Also add individual template paths
                    for field in template_fields:
                        if field in value:
                            template_path = f"{current_path}.{field}"
                            results.append({
                                "path": template_path,
                                "name": f"{current_path.replace('.', ' ').title()} - {field.replace('_', ' ').title()}",
                                "description": f"Config template: {template_path}",
                                "category": category
                            })
                else:
                    # Recurse into nested structure
                    self._extract_prompts_recursive(value, current_path, results)

    def _get_category_from_path(self, path: str) -> str:
        """Determine category from prompt path."""
        if "ontology" in path:
            return "ontology"
        elif "enricher" in path:
            return "enrichment"
        elif "search" in path:
            return "search"
        else:
            return "extraction"


# Global prompt provider instance for dependency injection
_prompt_provider: Optional[PromptProvider] = None


def set_prompt_provider(provider: PromptProvider) -> None:
    """Inject a prompt provider implementation.
    
    This allows different services to provide their own prompt implementations:
    - Smart-memory service: Uses ConfigPromptProvider (default)
    - Smart-memory-studio service: Can inject MongoPromptProvider
    """
    global _prompt_provider
    _prompt_provider = provider


def get_prompt_provider() -> PromptProvider:
    """Get the current prompt provider, defaulting to config-based."""
    global _prompt_provider
    if _prompt_provider is None:
        _prompt_provider = ConfigPromptProvider()
    return _prompt_provider


# Convenience functions that delegate to the injected provider
def get_prompt_value(path: str, default: Any = None) -> Any:
    """Get prompt value using the injected provider."""
    return get_prompt_provider().get_prompt_value(path, default)


def apply_placeholders(template: str, mapping: Dict[str, str]) -> str:
    """Apply placeholders using the injected provider."""
    return get_prompt_provider().apply_placeholders(template, mapping)


def reload_prompts() -> None:
    """Reload prompts using the injected provider."""
    get_prompt_provider().reload_prompts()
