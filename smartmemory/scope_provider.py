from typing import Dict, Any, Optional
from smartmemory.interfaces import ScopeProvider

class DefaultScopeProvider(ScopeProvider):
    """
    Default implementation that uses explicitly provided scope identifiers.
    Defaults to None (global/unscoped) if not provided.
    
    **OSS Usage**: When no parameters are provided (default), this allows unrestricted
    access to all data - perfect for single-user or development environments.
    
    Example OSS usage:
        # No scope provider = unrestricted access
        memory = SmartMemory()
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        team_id: Optional[str] = None
    ):
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.tenant_id = tenant_id
        self.team_id = team_id
    
    def get_isolation_filters(self) -> Dict[str, Any]:
        filters = {}
        
        # Org/Tenant (Hard Isolation)
        if self.tenant_id:
            filters["tenant_id"] = self.tenant_id
            
        # Workspace (Container Isolation)
        if self.workspace_id:
            filters["workspace_id"] = self.workspace_id
            
        # User (Owner Isolation)
        if self.user_id:
            filters["user_id"] = self.user_id
            
        return filters

    def get_write_context(self) -> Dict[str, Any]:
        # For the default provider, write context mirrors isolation filters
        # plus potentially team_id if available
        ctx = self.get_isolation_filters()
        
        # Add team_id if present (soft grouping)
        if self.team_id:
            ctx["team_id"] = self.team_id
            
        return ctx
