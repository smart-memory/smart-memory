"""
PromptHandler implementing BaseHandlerImpl[PromptTemplate] with domain logic and validation.
Provides synchronous prompt management with store delegation pattern and progressive fallback.
"""

import os
from datetime import datetime, timezone
from service_common.models.prompts import PromptTemplate, PromptCategory
from typing import Optional, Union, List, Any, Dict

from smartmemory.stores.base import BaseHandlerImpl


class PromptHandler(BaseHandlerImpl[PromptTemplate]):
    """Synchronous prompt handler implementing BaseHandlerImpl[PromptTemplate] with domain logic.
    
    Demonstrates:
    - Clean handler-store separation with progressive fallback
    - Store-agnostic handler implementation  
    - Domain logic (PromptTemplate) vs primitive storage (Dict)
    - Works with ANY BaseHandler[Dict] store (JSON, MongoDB, etc.)
    """

    def __init__(self, stores: Optional[Union[BaseHandlerImpl[Dict], List[BaseHandlerImpl[Dict]]]] = None):
        """Initialize handler with store(s) or config-based fallback.
        
        Args:
            stores: Single store, list of stores, or None for config-based loading
        """
        # Set domain type for automatic conversion
        self._domain_type = PromptTemplate
        # Let base class handle store initialization and fallback logic
        super().__init__(stores)

    def _load_stores_from_config(self) -> List[BaseHandlerImpl[Dict]]:
        """Load stores from configuration system."""
        # TODO: Replace with actual config system integration
        # This should read store configuration from config.json or similar
        # and instantiate the configured stores with their parameters

        # For now, raise an error to force explicit store provision
        # until proper config integration is implemented
        raise NotImplementedError(
            "Config-based store loading not yet implemented. "
            "Please provide explicit store(s) to PromptHandler constructor."
        )

    # Override add to include domain-specific validation
    def add(self, item: Union[PromptTemplate, Any], **kwargs) -> Union[str, PromptTemplate, None]:
        """Add a prompt template with domain logic and validation."""
        # Domain logic: Convert to PromptTemplate
        prompt = self._normalize_to_prompt_template(item)
        if not prompt:
            return None

        # Domain logic: Validation
        if not self._validate_prompt_template(prompt):
            return None

        # Base class handles the rest (conversion, store fallback)
        return super().add(prompt, **kwargs)

    # Override update to include domain-specific validation
    def update(self, item: Union[PromptTemplate, Any], **kwargs) -> Union[bool, PromptTemplate]:
        """Update a prompt template with domain logic."""
        # Domain logic: Convert to PromptTemplate
        prompt = self._normalize_to_prompt_template(item)
        if not prompt:
            return False

        # Domain logic: Validation
        if not prompt.id:
            return False

        # Base class handles the rest (conversion, store fallback)
        return super().update(prompt, **kwargs)

    # Domain logic helpers (prompt-specific business logic)

    def _normalize_to_prompt_template(self, item: Union[PromptTemplate, Any]) -> Optional[PromptTemplate]:
        """Convert various inputs to PromptTemplate - domain logic."""
        if isinstance(item, PromptTemplate):
            return item

        if hasattr(item, 'to_prompt_template'):
            return item.to_prompt_template()

        if isinstance(item, dict):
            try:
                return PromptTemplate.from_dict(item)
            except Exception:
                return None

        if isinstance(item, str):
            # Create minimal prompt from string content
            return PromptTemplate(
                name="auto_generated",
                template_content=item,
                category=PromptCategory.GENERAL
            )

        return None

    def _validate_prompt_template(self, prompt: PromptTemplate) -> bool:
        """Validate PromptTemplate - domain logic."""
        if not prompt:
            return False

        if not prompt.name or not prompt.name.strip():
            return False

        if not prompt.template_content or not prompt.template_content.strip():
            return False

        if not isinstance(prompt.category, PromptCategory):
            return False

        return True

    def _apply_prompt_defaults(self, prompt: PromptTemplate):
        """Apply domain-specific defaults."""
        if not prompt.created_at:
            prompt.created_at = datetime.now(timezone.utc)

        prompt.updated_at = datetime.now(timezone.utc)

        if prompt.usage_count is None:
            prompt.usage_count = 0

        if not prompt.version:
            prompt.version = "v1"

    def _dict_to_prompt_template(self, data: Dict) -> Optional[PromptTemplate]:
        """Convert primitive dict to PromptTemplate - domain logic."""
        try:
            return PromptTemplate.from_dict(data)
        except Exception:
            # Fallback: create PromptTemplate from dict keys
            return PromptTemplate(
                id=data.get('id'),
                name=data.get('name', ''),
                template_content=data.get('template_content', ''),
                category=PromptCategory(data.get('category', PromptCategory.GENERAL)),
                description=data.get('description'),
                version=data.get('version', 'v1'),
                workspace_id=data.get('workspace_id'),
                created_at=data.get('created_at'),
                updated_at=data.get('updated_at')
            )

    def _increment_version(self, current_version: str) -> str:
        """Increment version string (v1 -> v2, etc.)."""
        if current_version.startswith('v') and current_version[1:].isdigit():
            version_num = int(current_version[1:]) + 1
            return f"v{version_num}"
        else:
            return f"{current_version}_updated"

    def _log_prompt_operation(self, prompt: PromptTemplate, operation: str):
        """Log prompt operations for analytics."""
        # This would typically save to a usage log store
        # For now, we'll use the store's built-in analytics
        pass

    def _log_bulk_operation(self, operation: str, **kwargs):
        """Log bulk operations."""
        # Placeholder for bulk operation logging
        pass

    # Prompt-specific domain methods (not part of BaseHandler)

    def find_by_name(self, name: str, workspace_id: Optional[str] = None) -> Optional[PromptTemplate]:
        """Find prompt by name with workspace isolation."""
        results = self.search("", name=name, workspace_id=workspace_id, limit=1)
        return results[0] if results else None

    def find_by_category(self, category: PromptCategory, workspace_id: Optional[str] = None) -> List[PromptTemplate]:
        """Find prompts by category."""
        return self.search("", category=category.value, workspace_id=workspace_id, is_active=True)

    def find_active_prompts(self, workspace_id: Optional[str] = None) -> List[PromptTemplate]:
        """Get all active prompts."""
        return self.search("", is_active=True, workspace_id=workspace_id)

    def find_by_config_path(self, config_path: str, workspace_id: Optional[str] = None) -> Optional[PromptTemplate]:
        """Find prompt override by config path."""
        results = self.search("", config_path=config_path, workspace_id=workspace_id, is_active=True, limit=1)
        return results[0] if results else None

    def search_by_tags(self, tags: List[str], workspace_id: Optional[str] = None) -> List[PromptTemplate]:
        """Search prompts by tags."""
        # This is a simplified implementation - could be enhanced with better tag matching
        all_prompts = self.search("", workspace_id=workspace_id, is_active=True)

        matching_prompts = []
        for prompt in all_prompts:
            if prompt.tags and any(tag in prompt.tags for tag in tags):
                matching_prompts.append(prompt)

        return matching_prompts

    def increment_usage(self, prompt_id: str) -> bool:
        """Increment usage counter for a prompt."""
        prompt = self.get(prompt_id)
        if not prompt:
            return False

        prompt.increment_usage()
        result = self.update(prompt)
        return bool(result)

    def render_template(self, prompt_id: str, variables: Dict[str, Any]) -> Optional[str]:
        """Render a template with variables and track usage."""
        prompt = self.get(prompt_id)
        if not prompt:
            return None

        # Render template
        rendered = prompt.render(variables)

        # Increment usage
        self.increment_usage(prompt_id)

        return rendered

    def create_version(self, prompt_id: str, changes: Dict[str, Any]) -> Optional[PromptTemplate]:
        """Create a new version of an existing prompt."""
        original = self.get(prompt_id)
        if not original:
            return None

        # Create new version
        new_prompt = PromptTemplate.from_dict(original.to_dict())
        new_prompt.id = None  # Will get new ID
        new_prompt.version = self._increment_version(original.version)

        # Apply changes
        for key, value in changes.items():
            if hasattr(new_prompt, key):
                setattr(new_prompt, key, value)

        # Save new version
        result = self.add(new_prompt)
        return new_prompt if result else None

    def get_usage_analytics(self) -> Dict[str, Any]:
        """Get prompt usage analytics from first store with analytics."""
        for store in self.stores:
            if hasattr(store, 'get_usage_analytics'):
                return store.get_usage_analytics()
        return {}


# Factory functions - simple and clean

def create_prompt_handler(store: Union[BaseHandlerImpl[Dict], List[BaseHandlerImpl[Dict]]] = None) -> PromptHandler:
    """Create prompt handler.
    
    Args:
        store: Single store, list of stores, or None for default config-based loading
    """
    return PromptHandler(store)


def create_json_prompt_handler(base_directory: str = None, workspace_id: str = None) -> PromptHandler:
    """Create prompt handler with explicit JSON storage."""
    if base_directory is None:
        workspace = workspace_id or os.getenv("WORKSPACE_ID", "default")
        base_directory = os.path.expanduser(f"~/.smartmemory/workspaces/{workspace}/prompts")

    os.makedirs(base_directory, exist_ok=True)

    from smartmemory.stores.json_store import JSONStore
    json_store = JSONStore(
        base_directory=base_directory,
        use_case="prompts",
        auto_save=True
    )
    return PromptHandler(json_store)


def create_multi_store_prompt_handler(stores: List[BaseHandlerImpl[Dict]]) -> PromptHandler:
    """Create prompt handler with explicit list of stores for fallback."""
    return PromptHandler(stores)


# Example usage:
def example_usage():
    """Demonstrate the PromptHandler in action."""

    # Create handler with JSON storage
    handler = create_json_prompt_handler("./prompt_data")

    # Create a new prompt
    prompt = PromptTemplate(
        name="example_extraction",
        category=PromptCategory.EXTRACTION,
        template_content="Extract entities from: {text}",
        parameters={"text": "default text"},
        description="Example extraction prompt"
    )

    # Add prompt (async)
    prompt_id = handler.add_async(prompt)
    print(f"Created prompt: {prompt_id}")

    # Find by name
    found_prompt = handler.find_by_name_async("example_extraction")
    print(f"Found prompt: {found_prompt.name if found_prompt else 'None'}")

    # Render template
    rendered = handler.render_template_async(prompt_id, {"text": "John is a doctor"})
    print(f"Rendered: {rendered}")

    # Get usage analytics
    analytics = handler.get_usage_analytics()
    print(f"Analytics: {analytics}")

    return {
        'prompt': found_prompt,
        'rendered': rendered,
        'analytics': analytics
    }
