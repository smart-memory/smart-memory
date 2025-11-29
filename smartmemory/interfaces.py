from abc import ABC, abstractmethod
from typing import Dict, Any

class ScopeProvider(ABC):
    """
    Interface for providing isolation scope context to storage backends.
    Allows the service layer to define tenancy/policy, which the core library
    enforces transparently.
    """
    
    @abstractmethod
    def get_isolation_filters(self) -> Dict[str, Any]:
        """
        Return a dictionary of filters that MUST be applied to data queries.
        Example: {'tenant_id': 't1', 'workspace_id': 'w1'}
        Used for READ operations (MATCH, SEARCH).
        """
        pass
        
    @abstractmethod
    def get_write_context(self) -> Dict[str, Any]:
        """
        Return a dictionary of properties that MUST be written to new data.
        Example: {'tenant_id': 't1', 'created_by': 'u1'}
        Used for WRITE operations (CREATE, MERGE).
        """
        pass

    @abstractmethod
    def get_global_search_filters(self) -> Dict[str, Any]:
        """
        Return isolation filters for global/shared data searches.
        Excludes user-level isolation but keeps tenant/workspace boundaries.
        """
        pass

    @abstractmethod
    def get_user_isolation_key(self) -> str:
        """
        Return the field name used for user-level isolation.
        Allows backends to check for user-scoped vs global data without hardcoding.
        """
        pass
