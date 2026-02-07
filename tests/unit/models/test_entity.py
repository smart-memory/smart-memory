"""Unit tests for Entity model."""

from smartmemory.models.entity import Entity
from smartmemory.models.memory_item import MemoryItem


class TestEntityCreation:
    def test_defaults(self):
        entity = Entity()
        assert entity.name == ""
        assert entity.entity_type == ""
        assert entity.properties == {}
        assert entity.item_id is None

    def test_explicit_args(self):
        entity = Entity(
            name="Alice",
            entity_type="person",
            item_id="ent-123",
            properties={"age": 30, "role": "engineer"},
        )
        assert entity.name == "Alice"
        assert entity.entity_type == "person"
        assert entity.item_id == "ent-123"
        assert entity.properties == {"age": 30, "role": "engineer"}


class TestEntityToMemoryItem:
    def test_converts_to_memory_item(self):
        entity = Entity(
            name="Python",
            entity_type="technology",
            item_id="ent-py",
            properties={"paradigm": "multi"},
        )
        item = entity.to_memory_item()
        assert isinstance(item, MemoryItem)
        assert item.memory_type == "Entity"
        assert item.content == "Python"
        assert item.item_id == "ent-py"
        assert item.metadata["entity_type"] == "technology"
        assert item.metadata["properties"] == {"paradigm": "multi"}

    def test_generates_item_id_when_none(self):
        entity = Entity(name="Test")
        item = entity.to_memory_item()
        assert item.item_id is not None
        assert len(item.item_id) > 0


class TestEntityFromMemoryItem:
    def test_creates_entity_from_memory_item(self):
        item = MemoryItem(
            item_id="mem-1",
            content="Alice",
            metadata={
                "entity_type": "person",
                "properties": {"role": "admin"},
            },
        )
        entity = Entity.from_memory_item(item)
        assert entity.item_id == "mem-1"
        assert entity.name == "Alice"
        assert entity.entity_type == "person"
        assert entity.properties == {"role": "admin"}

    def test_falls_back_to_metadata_name(self):
        item = MemoryItem(
            item_id="mem-2",
            metadata={"name": "Bob", "entity_type": "person"},
        )
        entity = Entity.from_memory_item(item)
        assert entity.name == "Bob"

    def test_falls_back_to_type_key(self):
        item = MemoryItem(
            item_id="mem-3",
            content="Widget",
            metadata={"type": "product"},
        )
        entity = Entity.from_memory_item(item)
        assert entity.entity_type == "product"

    def test_handles_missing_properties(self):
        item = MemoryItem(item_id="mem-4", content="Thing")
        entity = Entity.from_memory_item(item)
        assert entity.properties == {}


class TestEntityRoundtrip:
    def test_entity_to_memory_item_and_back(self):
        original = Entity(
            name="SmartMemory",
            entity_type="project",
            item_id="ent-sm",
            properties={"lang": "python"},
        )
        item = original.to_memory_item()
        restored = Entity.from_memory_item(item)

        assert restored.name == original.name
        assert restored.entity_type == original.entity_type
        assert restored.item_id == original.item_id
        assert restored.properties == original.properties
