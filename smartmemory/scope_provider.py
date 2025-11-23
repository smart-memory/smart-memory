from typing import Dict, Any, Optional
from smartmemory.interfaces import ScopeProvider

class DefaultScopeProvider(ScopeProvider):
    """
    Default implementation that uses explicitly provided scope identifiers.
    defaults to None (global/unscoped) if not provided.
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        org_id: Optional[str] = None,
        team_id: Optional[str] = None
    ):
        self.user_id = user_id
        self.workspace_id = workspace_id
        self.org_id = org_id
        self.team_id = team_id
    
    def get_isolation_filters(self) -> Dict[str, Any]:
        filters = {}
        
        # Org/Tenant (Hard Isolation)
        if self.org_id:
            filters["tenant_id"] = self.org_id
            
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
