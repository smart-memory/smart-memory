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
