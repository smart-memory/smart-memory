"""
Token usage tracking for LLM operations.

token tracking, this module provides utilities
for tracking and reporting LLM token usage across operations.

Features:
- Track prompt and completion tokens
- Aggregate usage across multiple calls
- Cost estimation (optional)
- Thread-safe accumulation
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


@dataclass
class AggregatedUsage:
    """Aggregated token usage across multiple calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    models_used: Dict[str, int] = field(default_factory=dict)
    
    def add(self, usage: TokenUsage):
        """Add a single usage record."""
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.call_count += 1
        
        if usage.model:
            self.models_used[usage.model] = self.models_used.get(usage.model, 0) + 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "models_used": dict(self.models_used)
        }


class TokenTracker:
    """
    Thread-safe token usage tracker.
    
    Usage:
        tracker = TokenTracker()
        
        # Track usage from LLM response
        tracker.track(response)
        
        # Get totals
        usage = tracker.get_usage()
        print(f"Total tokens: {usage.total_tokens}")
        
        # Reset for new operation
        tracker.reset()
    """
    
    def __init__(self):
        """Initialize tracker."""
        self._lock = threading.Lock()
        self._usage = AggregatedUsage()
        self._history: List[TokenUsage] = []
        self._enabled = True
    
    def enable(self):
        """Enable tracking."""
        self._enabled = True
    
    def disable(self):
        """Disable tracking."""
        self._enabled = False
    
    def track(
        self,
        response: Any = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        model: str = ""
    ):
        """
        Track token usage.
        
        Can be called with:
        - A response object (will extract usage from it)
        - Explicit token counts
        
        Args:
            response: LLM response object (optional)
            prompt_tokens: Prompt token count
            completion_tokens: Completion token count
            total_tokens: Total token count
            model: Model name
        """
        if not self._enabled:
            return
        
        # Extract usage from response if provided
        if response is not None:
            extracted = self._extract_usage(response)
            if extracted:
                prompt_tokens = extracted.get('prompt_tokens', prompt_tokens)
                completion_tokens = extracted.get('completion_tokens', completion_tokens)
                total_tokens = extracted.get('total_tokens', total_tokens)
                model = extracted.get('model', model)
        
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=model
        )
        
        with self._lock:
            self._usage.add(usage)
            self._history.append(usage)
        
        logger.debug(f"Tracked {usage.total_tokens} tokens ({model})")
    
    def _extract_usage(self, response: Any) -> Optional[Dict[str, Any]]:
        """Extract usage from various response formats."""
        # Dict response
        if isinstance(response, dict):
            usage = response.get('usage')
            if usage:
                return {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'model': response.get('model', '')
                }
            
            # Check nested response
            nested = response.get('response', {})
            if isinstance(nested, dict):
                usage = nested.get('usage')
                if usage:
                    return {
                        'prompt_tokens': usage.get('prompt_tokens', 0),
                        'completion_tokens': usage.get('completion_tokens', 0),
                        'total_tokens': usage.get('total_tokens', 0),
                        'model': nested.get('model', '')
                    }
        
        # Object with usage attribute
        if hasattr(response, 'usage'):
            usage = response.usage
            if hasattr(usage, 'prompt_tokens'):
                return {
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0),
                    'total_tokens': getattr(usage, 'total_tokens', 0),
                    'model': getattr(response, 'model', '')
                }
        
        return None
    
    def get_usage(self) -> AggregatedUsage:
        """Get current aggregated usage."""
        with self._lock:
            return AggregatedUsage(
                prompt_tokens=self._usage.prompt_tokens,
                completion_tokens=self._usage.completion_tokens,
                total_tokens=self._usage.total_tokens,
                call_count=self._usage.call_count,
                models_used=dict(self._usage.models_used)
            )
    
    def get_history(self) -> List[TokenUsage]:
        """Get usage history."""
        with self._lock:
            return list(self._history)
    
    def reset(self):
        """Reset tracker."""
        with self._lock:
            self._usage = AggregatedUsage()
            self._history = []
    
    def __str__(self) -> str:
        usage = self.get_usage()
        return (
            f"TokenTracker: {usage.total_tokens} total tokens "
            f"({usage.prompt_tokens} prompt, {usage.completion_tokens} completion) "
            f"across {usage.call_count} calls"
        )


# Global tracker instance
_global_tracker = TokenTracker()


def get_global_tracker() -> TokenTracker:
    """Get the global token tracker."""
    return _global_tracker


def track_usage(
    response: Any = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    model: str = ""
):
    """Track usage on the global tracker."""
    _global_tracker.track(
        response=response,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        model=model
    )


def get_usage() -> Dict[str, Any]:
    """Get usage from global tracker as dict."""
    return _global_tracker.get_usage().to_dict()


def reset_usage():
    """Reset global tracker."""
    _global_tracker.reset()


# Cost estimation (approximate, may vary)
COST_PER_1K_TOKENS = {
    "gpt-4o": {"prompt": 0.005, "completion": 0.015},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-5": {"prompt": 0.01, "completion": 0.03},
    "gpt-5-mini": {"prompt": 0.001, "completion": 0.003},
    "default": {"prompt": 0.001, "completion": 0.002}
}


def estimate_cost(usage: AggregatedUsage, model: str = "default") -> float:
    """
    Estimate cost for token usage.
    
    Args:
        usage: Aggregated usage
        model: Model name for pricing
        
    Returns:
        Estimated cost in USD
    """
    # Find pricing
    pricing = COST_PER_1K_TOKENS.get(model)
    if not pricing:
        # Try to match by prefix
        for key in COST_PER_1K_TOKENS:
            if model.startswith(key):
                pricing = COST_PER_1K_TOKENS[key]
                break
    
    if not pricing:
        pricing = COST_PER_1K_TOKENS["default"]
    
    prompt_cost = (usage.prompt_tokens / 1000) * pricing["prompt"]
    completion_cost = (usage.completion_tokens / 1000) * pricing["completion"]
    
    return prompt_cost + completion_cost
