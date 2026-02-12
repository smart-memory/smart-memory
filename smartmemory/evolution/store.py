"""
Evolution event store for MongoDB persistence.

Provides CRUD operations for evolution events, including
retrieval of event history and confidence trajectory data.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from smartmemory.evolution.models import EvolutionEvent

logger = logging.getLogger(__name__)

COLLECTION_NAME = "procedure_evolution_events"


class EvolutionEventStore:
    """MongoDB store for procedure evolution events.

    Provides methods for creating, retrieving, and querying evolution events
    with proper indexing for efficient timeline and trajectory queries.
    """

    def __init__(self, db: Database):
        """Initialize the store with a MongoDB database.

        Args:
            db: MongoDB database instance
        """
        self.db = db
        self.collection: Collection = db[COLLECTION_NAME]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes for efficient queries."""
        try:
            # Primary timeline query: get events for a procedure sorted by time
            self.collection.create_index(
                [
                    ("workspace_id", ASCENDING),
                    ("procedure_id", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="workspace_procedure_timeline",
            )

            # Event type query: filter by event type within a workspace
            self.collection.create_index(
                [
                    ("workspace_id", ASCENDING),
                    ("event_type", ASCENDING),
                    ("timestamp", DESCENDING),
                ],
                name="workspace_event_type",
            )

            # Version lookup: get specific version of a procedure
            self.collection.create_index(
                [
                    ("procedure_id", ASCENDING),
                    ("version", ASCENDING),
                ],
                name="procedure_version",
                unique=True,
            )

            # Event ID lookup
            self.collection.create_index(
                [("event_id", ASCENDING)],
                name="event_id",
                unique=True,
            )

            logger.debug("Evolution event indexes ensured")
        except Exception as e:
            logger.warning("Failed to create evolution event indexes: %s", e)

    def save(self, event: EvolutionEvent) -> str:
        """Save an evolution event to the store.

        Args:
            event: The evolution event to save

        Returns:
            The event_id of the saved event

        Raises:
            Exception: If save fails
        """
        doc = event.to_dict()
        try:
            self.collection.insert_one(doc)
            logger.debug("Saved evolution event %s for procedure %s", event.event_id, event.procedure_id)
            return event.event_id
        except Exception as e:
            logger.error("Failed to save evolution event: %s", e)
            raise

    def get_by_id(self, event_id: str) -> Optional[EvolutionEvent]:
        """Get an evolution event by its ID.

        Args:
            event_id: The event ID to look up

        Returns:
            The EvolutionEvent or None if not found
        """
        doc = self.collection.find_one({"event_id": event_id})
        if doc:
            return EvolutionEvent.from_dict(doc)
        return None

    def get_by_procedure_and_event(
        self,
        procedure_id: str,
        event_id: str,
        workspace_id: Optional[str] = None,
    ) -> Optional[EvolutionEvent]:
        """Get an evolution event by procedure ID and event ID.

        Args:
            procedure_id: The procedure ID
            event_id: The event ID
            workspace_id: Optional workspace ID for tenant isolation

        Returns:
            The EvolutionEvent or None if not found
        """
        query: Dict[str, Any] = {
            "procedure_id": procedure_id,
            "event_id": event_id,
        }
        if workspace_id:
            query["workspace_id"] = workspace_id

        doc = self.collection.find_one(query)
        if doc:
            return EvolutionEvent.from_dict(doc)
        return None

    def get_history(
        self,
        procedure_id: str,
        workspace_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[EvolutionEvent]:
        """Get evolution history for a procedure.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID for tenant isolation
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List of EvolutionEvents, newest first
        """
        cursor = (
            self.collection.find(
                {
                    "workspace_id": workspace_id,
                    "procedure_id": procedure_id,
                }
            )
            .sort("timestamp", DESCENDING)
            .skip(offset)
            .limit(limit)
        )

        return [EvolutionEvent.from_dict(doc) for doc in cursor]

    def get_latest_version(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> int:
        """Get the latest version number for a procedure.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            The latest version number, or 0 if no events exist
        """
        doc = self.collection.find_one(
            {
                "workspace_id": workspace_id,
                "procedure_id": procedure_id,
            },
            sort=[("version", DESCENDING)],
            projection={"version": 1},
        )

        if doc:
            return doc.get("version", 0)
        return 0

    def get_previous_event(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> Optional[EvolutionEvent]:
        """Get the most recent evolution event for a procedure.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            The most recent EvolutionEvent or None
        """
        doc = self.collection.find_one(
            {
                "workspace_id": workspace_id,
                "procedure_id": procedure_id,
            },
            sort=[("timestamp", DESCENDING)],
        )

        if doc:
            return EvolutionEvent.from_dict(doc)
        return None

    def get_confidence_trajectory(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> List[Dict[str, Any]]:
        """Get confidence trajectory data for charting.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            List of data points with timestamp, confidence, matches, and success_rate
        """
        cursor = self.collection.find(
            {
                "workspace_id": workspace_id,
                "procedure_id": procedure_id,
            },
            projection={
                "timestamp": 1,
                "confidence_at_event": 1,
                "match_stats_at_event": 1,
                "event_type": 1,
                "version": 1,
            },
        ).sort("timestamp", ASCENDING)

        data_points = []
        for doc in cursor:
            timestamp = doc.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp_str = timestamp.isoformat()
            else:
                timestamp_str = str(timestamp) if timestamp else None

            match_stats = doc.get("match_stats_at_event", {})
            data_points.append(
                {
                    "timestamp": timestamp_str,
                    "confidence": doc.get("confidence_at_event", 0.0),
                    "matches": match_stats.get("total_matches", 0),
                    "success_rate": match_stats.get("success_rate", 0.0),
                    "event_type": doc.get("event_type"),
                    "version": doc.get("version"),
                }
            )

        return data_points

    def count_events(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> int:
        """Count total evolution events for a procedure.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            Total count of events
        """
        return self.collection.count_documents(
            {
                "workspace_id": workspace_id,
                "procedure_id": procedure_id,
            }
        )

    def delete_events_for_procedure(
        self,
        procedure_id: str,
        workspace_id: str,
    ) -> int:
        """Delete all evolution events for a procedure.

        Args:
            procedure_id: The procedure ID
            workspace_id: The workspace ID

        Returns:
            Number of deleted events
        """
        result = self.collection.delete_many(
            {
                "workspace_id": workspace_id,
                "procedure_id": procedure_id,
            }
        )
        return result.deleted_count
