"""Tests for LayeredOntologyService."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from smartmemory.ontology.layer_service import LayeredOntologyService
from smartmemory.ontology.models import (
    EntityTypeDefinition,
    Ontology,
    OntologySubscription,
)


def _make_entity(name: str) -> EntityTypeDefinition:
    return EntityTypeDefinition(
        name=name,
        description="test",
        properties={},
        required_properties=set(),
        parent_types=set(),
        aliases=set(),
        examples=[],
        created_by="test",
        created_at=datetime.now(),
    )


def _make_ontology(name: str, ontology_id: str = "test-id", tenant_id: str = "t1") -> Ontology:
    o = Ontology(name=name)
    o.id = ontology_id
    o.tenant_id = tenant_id
    return o


class TestLoadLayered:
    def test_without_subscription(self):
        overlay = _make_ontology("overlay", "ov-id")
        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        layered = svc.load_layered("ov-id")
        assert layered.overlay is overlay
        assert layered.base is None

    def test_with_subscription(self):
        base = _make_ontology("base", "base-id")
        base.add_entity_type(_make_entity("Animal"))
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="base-id")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: {"ov-id": overlay, "base-id": base}.get(oid)
        svc = LayeredOntologyService(storage)

        layered = svc.load_layered("ov-id")
        assert layered.base is base
        assert "animal" in layered.entity_types

    def test_base_not_found_logs_warning(self, caplog):
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="missing-base")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: overlay if oid == "ov-id" else None
        svc = LayeredOntologyService(storage)

        with caplog.at_level("WARNING"):
            layered = svc.load_layered("ov-id")
        assert layered.base is None
        assert "not found" in caplog.text

    def test_overlay_not_found_raises(self):
        storage = MagicMock()
        storage.load_ontology.return_value = None
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="not found"):
            svc.load_layered("missing")


class TestSubscribe:
    def test_subscribe_success(self):
        base = _make_ontology("base", "base-id", tenant_id="t1")
        overlay = _make_ontology("overlay", "ov-id", tenant_id="t1")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: {"ov-id": overlay, "base-id": base}.get(oid)
        svc = LayeredOntologyService(storage)

        svc.subscribe("ov-id", "base-id", "t1", "user1")

        assert overlay.subscription is not None
        assert overlay.subscription.base_registry_id == "base-id"
        assert overlay.subscription.subscribed_by == "user1"
        storage.save_ontology.assert_called_once_with(overlay)

    def test_subscribe_already_subscribed(self):
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="old-base")

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="already subscribed"):
            svc.subscribe("ov-id", "new-base", "t1", "user1")

    def test_subscribe_base_not_found(self):
        overlay = _make_ontology("overlay", "ov-id")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: overlay if oid == "ov-id" else None
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="Base ontology not found"):
            svc.subscribe("ov-id", "missing-base", "t1", "user1")

    def test_subscribe_cross_tenant_rejected(self):
        base = _make_ontology("base", "base-id", tenant_id="other-tenant")
        overlay = _make_ontology("overlay", "ov-id", tenant_id="t1")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: {"ov-id": overlay, "base-id": base}.get(oid)
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="different tenant"):
            svc.subscribe("ov-id", "base-id", "t1", "user1")

    def test_subscribe_overlay_not_found(self):
        storage = MagicMock()
        storage.load_ontology.return_value = None
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="not found"):
            svc.subscribe("missing", "base-id", "t1", "user1")

    def test_subscribe_system_base_allowed(self):
        """System bases (empty tenant_id) should be subscribable by any tenant."""
        base = _make_ontology("base", "base-id", tenant_id="")
        overlay = _make_ontology("overlay", "ov-id", tenant_id="t1")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: {"ov-id": overlay, "base-id": base}.get(oid)
        svc = LayeredOntologyService(storage)

        svc.subscribe("ov-id", "base-id", "t1", "user1")
        assert overlay.subscription is not None

    def test_subscribe_with_pinned_version(self):
        base = _make_ontology("base", "base-id", tenant_id="t1")
        overlay = _make_ontology("overlay", "ov-id", tenant_id="t1")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: {"ov-id": overlay, "base-id": base}.get(oid)
        svc = LayeredOntologyService(storage)

        svc.subscribe("ov-id", "base-id", "t1", "user1", pinned_version="2.0.0")
        assert overlay.subscription.pinned_version == "2.0.0"


class TestUnsubscribe:
    def test_unsubscribe_flattens(self):
        base = _make_ontology("base", "base-id")
        base.add_entity_type(_make_entity("Animal"))
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="base-id")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: {"ov-id": overlay, "base-id": base}.get(oid)
        svc = LayeredOntologyService(storage)

        svc.unsubscribe("ov-id")
        saved = storage.save_ontology.call_args[0][0]
        assert saved.subscription is None
        assert "animal" in saved.entity_types

    def test_unsubscribe_not_subscribed(self):
        overlay = _make_ontology("overlay", "ov-id")

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="not subscribed"):
            svc.unsubscribe("ov-id")


class TestPinUnpin:
    def test_pin_version(self):
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="base-id")

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        svc.pin_version("ov-id", "3.0.0", "user1")
        assert overlay.subscription.pinned_version == "3.0.0"
        storage.save_ontology.assert_called_once()

    def test_unpin(self):
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="base-id", pinned_version="2.0.0")

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        svc.unpin("ov-id")
        assert overlay.subscription.pinned_version is None
        storage.save_ontology.assert_called_once()

    def test_pin_not_subscribed(self):
        overlay = _make_ontology("overlay", "ov-id")

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="not subscribed"):
            svc.pin_version("ov-id", "1.0.0", "user1")


class TestHideUnhide:
    def test_hide_type(self):
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="base-id")

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        svc.hide_type("ov-id", "Animal")
        assert "animal" in overlay.subscription.hidden_types
        storage.save_ontology.assert_called_once()

    def test_unhide_type(self):
        overlay = _make_ontology("overlay", "ov-id")
        overlay.subscription = OntologySubscription(base_registry_id="base-id", hidden_types={"animal"})

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        svc.unhide_type("ov-id", "Animal")
        assert "animal" not in overlay.subscription.hidden_types
        storage.save_ontology.assert_called_once()

    def test_hide_not_subscribed(self):
        overlay = _make_ontology("overlay", "ov-id")

        storage = MagicMock()
        storage.load_ontology.return_value = overlay
        svc = LayeredOntologyService(storage)

        with pytest.raises(ValueError, match="not subscribed"):
            svc.hide_type("ov-id", "Animal")


class TestGetDiff:
    def test_diff_with_base(self):
        base = _make_ontology("base", "base-id")
        base.add_entity_type(_make_entity("Animal"))
        base.add_entity_type(_make_entity("Plant"))
        overlay = _make_ontology("overlay", "ov-id")
        overlay.add_entity_type(_make_entity("Plant"))
        overlay.add_entity_type(_make_entity("Robot"))
        overlay.subscription = OntologySubscription(base_registry_id="base-id")

        storage = MagicMock()
        storage.load_ontology.side_effect = lambda oid: {"ov-id": overlay, "base-id": base}.get(oid)
        svc = LayeredOntologyService(storage)

        diff = svc.get_diff("ov-id")
        assert diff.base_only == ["animal"]
        assert diff.overlay_only == ["robot"]
        assert diff.overridden == ["plant"]
