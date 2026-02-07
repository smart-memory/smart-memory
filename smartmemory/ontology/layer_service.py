"""Orchestrates the ontology subscription lifecycle using OntologyStorage."""

import logging
from datetime import datetime
from typing import Optional

from smartmemory.ontology.layered import LayeredOntology
from smartmemory.ontology.models import LayerDiff, OntologySubscription
from smartmemory.stores.ontology import OntologyStorage

logger = logging.getLogger(__name__)


class LayeredOntologyService:
    """Manages subscribing an overlay ontology to a base, hiding types, pinning versions, and detaching."""

    def __init__(self, storage: OntologyStorage) -> None:
        self.storage = storage

    def load_layered(self, registry_id: str) -> LayeredOntology:
        """Load overlay and resolve its subscription to produce a merged view.

        If the overlay has no subscription, returns a LayeredOntology with base=None.
        If the base cannot be found, logs a warning and returns base=None.
        """
        overlay = self.storage.load_ontology(registry_id)
        if overlay is None:
            raise ValueError(f"Ontology not found: {registry_id}")

        if overlay.subscription is None:
            return LayeredOntology(overlay=overlay)

        sub = overlay.subscription
        base = self.storage.load_ontology(sub.base_registry_id)
        if base is None:
            logger.warning(
                "Base ontology %s not found for overlay %s — returning overlay-only view",
                sub.base_registry_id,
                registry_id,
            )
            return LayeredOntology(overlay=overlay)

        return LayeredOntology(
            overlay=overlay,
            base=base,
            pinned_version=sub.pinned_version,
            hidden_types=sub.hidden_types,
        )

    def subscribe(
        self,
        registry_id: str,
        base_registry_id: str,
        tenant_id: str,
        user_id: str,
        pinned_version: Optional[str] = None,
    ) -> None:
        """Subscribe overlay to a base ontology.

        Raises ValueError if already subscribed, base not found, or cross-tenant.
        """
        overlay = self.storage.load_ontology(registry_id)
        if overlay is None:
            raise ValueError(f"Ontology not found: {registry_id}")
        if overlay.subscription is not None:
            raise ValueError(f"Ontology {registry_id} is already subscribed to {overlay.subscription.base_registry_id}")

        base = self.storage.load_ontology(base_registry_id)
        if base is None:
            raise ValueError(f"Base ontology not found: {base_registry_id}")
        # Allow system bases (empty tenant_id) or same-tenant bases
        if base.tenant_id and base.tenant_id != tenant_id:
            raise ValueError("Cannot subscribe to base ontology from a different tenant")

        overlay.subscription = OntologySubscription(
            base_registry_id=base_registry_id,
            pinned_version=pinned_version,
            subscribed_at=datetime.now(),
            subscribed_by=user_id,
        )
        overlay.updated_at = datetime.now()
        self.storage.save_ontology(overlay)

    def unsubscribe(self, registry_id: str) -> None:
        """Detach from base: flatten visible base types into overlay, remove subscription."""
        layered = self.load_layered(registry_id)
        if layered.overlay.subscription is None:
            raise ValueError(f"Ontology {registry_id} is not subscribed to any base")

        flat = layered.detach()
        self.storage.save_ontology(flat)

    def pin_version(self, registry_id: str, version: str, user_id: str) -> None:
        """Pin subscription to a specific base version."""
        overlay = self.storage.load_ontology(registry_id)
        if overlay is None:
            raise ValueError(f"Ontology not found: {registry_id}")
        if overlay.subscription is None:
            raise ValueError(f"Ontology {registry_id} is not subscribed to any base")

        overlay.subscription.pinned_version = version
        overlay.updated_at = datetime.now()
        self.storage.save_ontology(overlay)

    def unpin(self, registry_id: str) -> None:
        """Clear pinned version — follow latest base."""
        overlay = self.storage.load_ontology(registry_id)
        if overlay is None:
            raise ValueError(f"Ontology not found: {registry_id}")
        if overlay.subscription is None:
            raise ValueError(f"Ontology {registry_id} is not subscribed to any base")

        overlay.subscription.pinned_version = None
        overlay.updated_at = datetime.now()
        self.storage.save_ontology(overlay)

    def hide_type(self, registry_id: str, type_name: str) -> None:
        """Add a type to the hidden set on the subscription."""
        overlay = self.storage.load_ontology(registry_id)
        if overlay is None:
            raise ValueError(f"Ontology not found: {registry_id}")
        if overlay.subscription is None:
            raise ValueError(f"Ontology {registry_id} is not subscribed to any base")

        overlay.subscription.hidden_types.add(type_name.lower())
        overlay.updated_at = datetime.now()
        self.storage.save_ontology(overlay)

    def unhide_type(self, registry_id: str, type_name: str) -> None:
        """Remove a type from the hidden set on the subscription."""
        overlay = self.storage.load_ontology(registry_id)
        if overlay is None:
            raise ValueError(f"Ontology not found: {registry_id}")
        if overlay.subscription is None:
            raise ValueError(f"Ontology {registry_id} is not subscribed to any base")

        overlay.subscription.hidden_types.discard(type_name.lower())
        overlay.updated_at = datetime.now()
        self.storage.save_ontology(overlay)

    def get_diff(self, registry_id: str) -> LayerDiff:
        """Return the diff between base and overlay layers."""
        layered = self.load_layered(registry_id)
        return layered.compute_diff()
