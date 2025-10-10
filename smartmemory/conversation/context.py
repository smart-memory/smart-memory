# Copyright (C) 2025 SmartMemory
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
# 
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# For commercial licensing options, please contact: help@smartmemory.ai
# Commercial licenses are available for organizations that wish to use
# this software in proprietary applications without the AGPL restrictions.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ConversationContext:
    """
    Lightweight conversation context used to bias retrieval/enrichment.
    This is additive and optional. All fields use snake_case.
    """

    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    topics: List[str] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)  # {name, id?, type?}
    sentiment: Optional[str] = None
    turn_history: List[Dict[str, Any]] = field(default_factory=list)  # [{role, content, ts}]
    active_threads: List[str] = field(default_factory=list)

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            "last_updated_at": self.last_updated_at.isoformat() if isinstance(self.last_updated_at, datetime) else self.last_updated_at,
            "topics": list(self.topics or []),
            "entities": list(self.entities or []),
            "sentiment": self.sentiment,
            "turn_history": list(self.turn_history or []),
            "active_threads": list(self.active_threads or []),
            "extra": dict(self.extra or {}),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        if data is None:
            return cls()
        ctx = cls(
            conversation_id=data.get("conversation_id"),
            user_id=data.get("user_id"),
            topics=list(data.get("topics") or []),
            entities=list(data.get("entities") or []),
            sentiment=data.get("sentiment"),
            turn_history=list(data.get("turn_history") or []),
            active_threads=list(data.get("active_threads") or []),
            extra=dict(data.get("extra") or {}),
        )
        return ctx
