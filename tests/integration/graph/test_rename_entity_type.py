"""Integration tests for rename_entity_type and merge_entity_types functionality."""

import pytest
from uuid import uuid4

pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestRenameEntityType:
    """Tests for renaming entity types."""

    def test_rename_updates_all_matching_entities(self, real_smartmemory_for_integration):
        """Renaming an entity type updates all matching entities."""
        memory = real_smartmemory_for_integration

        # Create unique type names for this test
        old_type = f"Framework_{uuid4().hex[:8]}"
        new_type = f"Library_{uuid4().hex[:8]}"

        # Add entities with the old type via graph directly
        for i in range(3):
            memory._graph.add_node(
                item_id=f"entity_{uuid4()}",
                properties={
                    "content": f"Entity {i} with old type",
                    "entity_type": old_type,
                    "name": f"TestEntity{i}",
                },
                memory_type="semantic",
            )

        # Rename the entity type
        updated = memory.rename_entity_type(old_type, new_type)

        # Should have updated 3 entities
        assert updated == 3

    def test_rename_leaves_other_types_unchanged(self, real_smartmemory_for_integration):
        """Renaming one entity type does not affect other types."""
        memory = real_smartmemory_for_integration

        # Create unique type names
        type_a = f"TypeA_{uuid4().hex[:8]}"
        type_b = f"TypeB_{uuid4().hex[:8]}"
        type_a_new = f"TypeANew_{uuid4().hex[:8]}"

        # Add entities of both types
        entity_a_id = f"entity_a_{uuid4()}"
        entity_b_id = f"entity_b_{uuid4()}"

        memory._graph.add_node(
            item_id=entity_a_id,
            properties={"content": "Entity A", "entity_type": type_a, "name": "EntityA"},
            memory_type="semantic",
        )
        memory._graph.add_node(
            item_id=entity_b_id,
            properties={"content": "Entity B", "entity_type": type_b, "name": "EntityB"},
            memory_type="semantic",
        )

        # Rename only type_a
        updated = memory.rename_entity_type(type_a, type_a_new)

        assert updated == 1

        # Verify type_b entity is unchanged
        entity_b = memory._graph.get_node(entity_b_id)
        if entity_b:
            # Check the entity_type property
            props = entity_b.metadata if hasattr(entity_b, "metadata") else entity_b
            if isinstance(props, dict):
                assert props.get("entity_type") == type_b

    def test_rename_returns_zero_for_nonexistent_type(self, real_smartmemory_for_integration):
        """Renaming a non-existent type returns 0."""
        memory = real_smartmemory_for_integration

        updated = memory.rename_entity_type(f"NonExistent_{uuid4()}", "NewType")

        assert updated == 0

    def test_rename_is_idempotent(self, real_smartmemory_for_integration):
        """Renaming the same type twice is safe."""
        memory = real_smartmemory_for_integration

        old_type = f"OldType_{uuid4().hex[:8]}"
        new_type = f"NewType_{uuid4().hex[:8]}"

        # Add an entity
        memory._graph.add_node(
            item_id=f"entity_{uuid4()}",
            properties={"content": "Test entity", "entity_type": old_type, "name": "TestEntity"},
            memory_type="semantic",
        )

        # First rename
        first_update = memory.rename_entity_type(old_type, new_type)
        assert first_update == 1

        # Second rename (old_type no longer exists)
        second_update = memory.rename_entity_type(old_type, new_type)
        assert second_update == 0


class TestMergeEntityTypes:
    """Tests for merging multiple entity types."""

    def test_merge_combines_multiple_types(self, real_smartmemory_for_integration):
        """Merging multiple types into one updates all source entities."""
        memory = real_smartmemory_for_integration

        # Create unique type names
        type_1 = f"Type1_{uuid4().hex[:8]}"
        type_2 = f"Type2_{uuid4().hex[:8]}"
        target_type = f"Target_{uuid4().hex[:8]}"

        # Add entities of different types
        memory._graph.add_node(
            item_id=f"entity_{uuid4()}",
            properties={"content": "Entity from type 1", "entity_type": type_1, "name": "E1"},
            memory_type="semantic",
        )
        memory._graph.add_node(
            item_id=f"entity_{uuid4()}",
            properties={"content": "Entity from type 1 again", "entity_type": type_1, "name": "E1a"},
            memory_type="semantic",
        )
        memory._graph.add_node(
            item_id=f"entity_{uuid4()}",
            properties={"content": "Entity from type 2", "entity_type": type_2, "name": "E2"},
            memory_type="semantic",
        )

        # Merge both types into target
        total_updated = memory.merge_entity_types([type_1, type_2], target_type)

        # Should have updated 3 entities total (2 from type_1, 1 from type_2)
        assert total_updated == 3

    def test_merge_skips_target_type_in_sources(self, real_smartmemory_for_integration):
        """Merging does not modify entities already at target type."""
        memory = real_smartmemory_for_integration

        source_type = f"Source_{uuid4().hex[:8]}"
        target_type = f"Target_{uuid4().hex[:8]}"

        # Add entity with source type
        memory._graph.add_node(
            item_id=f"entity_{uuid4()}",
            properties={"content": "Source entity", "entity_type": source_type, "name": "Src"},
            memory_type="semantic",
        )

        # Add entity already at target type
        memory._graph.add_node(
            item_id=f"entity_{uuid4()}",
            properties={"content": "Target entity", "entity_type": target_type, "name": "Tgt"},
            memory_type="semantic",
        )

        # Merge including target_type in sources
        total_updated = memory.merge_entity_types([source_type, target_type], target_type)

        # Should only update 1 (source_type), not the one already at target
        assert total_updated == 1

    def test_merge_empty_source_list(self, real_smartmemory_for_integration):
        """Merging with empty source list returns 0."""
        memory = real_smartmemory_for_integration

        total_updated = memory.merge_entity_types([], "AnyTarget")

        assert total_updated == 0

    def test_merge_returns_cumulative_count(self, real_smartmemory_for_integration):
        """Merge returns the total count of all renamed entities."""
        memory = real_smartmemory_for_integration

        types = [f"Type{i}_{uuid4().hex[:8]}" for i in range(3)]
        target = f"MergedTarget_{uuid4().hex[:8]}"

        # Add 2 entities of each type (6 total)
        for entity_type in types:
            for j in range(2):
                memory._graph.add_node(
                    item_id=f"entity_{uuid4()}",
                    properties={
                        "content": f"Entity {j} of type {entity_type}",
                        "entity_type": entity_type,
                        "name": f"E_{entity_type}_{j}",
                    },
                    memory_type="semantic",
                )

        total_updated = memory.merge_entity_types(types, target)

        # Should be 6 (2 entities * 3 types)
        assert total_updated == 6
