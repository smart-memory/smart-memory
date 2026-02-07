"""Unit tests for MemoryItem model."""

import hashlib
import uuid

import pytest

from smartmemory.models.memory_item import MemoryItem, MEMORY_TYPES


class TestMemoryItemCreation:
    def test_defaults(self):
        item = MemoryItem()
        assert item.memory_type == "semantic"
        assert item.content == ""
        assert item.metadata == {}
        assert item.embedding is None
        assert item.entities is None
        assert item.relations is None
        assert item.group_id is None
        # item_id should be a valid UUID
        uuid.UUID(item.item_id)

    def test_explicit_args(self):
        item = MemoryItem(
            content="hello world",
            memory_type="episodic",
            item_id="custom-id",
            group_id="grp-1",
            embedding=[0.1, 0.2, 0.3],
            metadata={"key": "val"},
        )
        assert item.content == "hello world"
        assert item.memory_type == "episodic"
        assert item.item_id == "custom-id"
        assert item.group_id == "grp-1"
        assert item.embedding == [0.1, 0.2, 0.3]
        assert item.metadata["key"] == "val"

    def test_type_alias_for_memory_type(self):
        item = MemoryItem(type="procedural")
        assert item.memory_type == "procedural"

    def test_metadata_defaults_to_empty_dict(self):
        item = MemoryItem()
        assert isinstance(item.metadata, dict)
        assert len(item.metadata) == 0


class TestMemoryTypes:
    def test_contains_all_nine_types(self):
        expected = {
            "semantic",
            "episodic",
            "procedural",
            "working",
            "zettel",
            "reasoning",
            "opinion",
            "observation",
            "decision",
        }
        assert MEMORY_TYPES == expected

    def test_count(self):
        assert len(MEMORY_TYPES) == 9


class TestMemoryItemSerialization:
    def test_to_dict_has_memory_type_key(self):
        item = MemoryItem(memory_type="working", content="test")
        d = item.to_dict()
        assert isinstance(d, dict)
        assert d["memory_type"] == "working"
        assert d["content"] == "test"

    def test_to_dict_excludes_immutable_fields_tracker(self):
        item = MemoryItem()
        d = item.to_dict()
        assert "_immutable_fields" not in d

    def test_to_vector_store_keys(self):
        item = MemoryItem(
            item_id="vec-1",
            content="some content",
            embedding=[1.0, 2.0],
        )
        vs = item.to_vector_store()
        assert set(vs.keys()) == {"id", "content", "vector", "metadata"}
        assert vs["id"] == "vec-1"
        assert vs["content"] == "some content"
        assert vs["vector"] == [1.0, 2.0]
        assert isinstance(vs["metadata"], dict)

    def test_to_vector_store_metadata_includes_memory_type(self):
        item = MemoryItem(memory_type="zettel", content="note")
        vs = item.to_vector_store()
        assert vs["metadata"]["memory_type"] == "zettel"


class TestMemoryItemClassMethods:
    def test_from_content_uses_content_hash_as_id(self):
        content = "test content for hashing"
        item = MemoryItem.from_content(content)
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert item.item_id == expected_hash
        assert item.content == content

    def test_from_content_respects_explicit_item_id(self):
        item = MemoryItem.from_content("stuff", item_id="my-id")
        assert item.item_id == "my-id"

    def test_from_content_passes_kwargs(self):
        item = MemoryItem.from_content("stuff", memory_type="episodic")
        assert item.memory_type == "episodic"


class TestMemoryItemStaticMethods:
    def test_cosine_similarity_identical_vectors(self):
        vec = [1.0, 0.0, 0.0]
        assert MemoryItem.cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        assert MemoryItem.cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self):
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        assert MemoryItem.cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_cosine_similarity_empty_vectors(self):
        assert MemoryItem.cosine_similarity([], []) == 0.0

    def test_cosine_similarity_mismatched_lengths(self):
        assert MemoryItem.cosine_similarity([1.0], [1.0, 2.0]) == 0.0

    def test_text_to_dummy_embedding_default_dim(self):
        emb = MemoryItem.text_to_dummy_embedding("hello")
        assert isinstance(emb, list)
        assert len(emb) == 32
        assert all(isinstance(v, float) for v in emb)

    def test_text_to_dummy_embedding_custom_dim(self):
        emb = MemoryItem.text_to_dummy_embedding("hello", dim=16)
        assert len(emb) == 16

    def test_text_to_dummy_embedding_deterministic(self):
        emb1 = MemoryItem.text_to_dummy_embedding("same text")
        emb2 = MemoryItem.text_to_dummy_embedding("same text")
        assert emb1 == emb2

    def test_compute_content_hash_returns_sha256(self):
        content = "hash me"
        result = MemoryItem.compute_content_hash(content)
        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert result == expected
        assert len(result) == 64  # SHA256 hex digest length

    def test_compute_content_hash_none_returns_none(self):
        assert MemoryItem.compute_content_hash(None) is None


class TestMemoryItemImmutability:
    def test_content_immutable_after_set(self):
        item = MemoryItem(content="original")
        with pytest.raises(ValueError, match="content has already been set"):
            item.content = "modified"

    def test_embedding_immutable_after_set(self):
        item = MemoryItem(embedding=[1.0, 2.0])
        with pytest.raises(ValueError, match="embedding has already been set"):
            item.embedding = [3.0, 4.0]

    def test_content_allows_setting_when_empty(self):
        item = MemoryItem()
        # Empty string is falsy, so re-setting is allowed
        item.content = "now set"
        assert item.content == "now set"

    def test_embedding_allows_setting_when_none(self):
        item = MemoryItem()
        item.embedding = [1.0]
        assert item.embedding == [1.0]

    def test_same_value_is_idempotent(self):
        item = MemoryItem(content="hello")
        # Setting to the same value should not raise
        item.content = "hello"
        assert item.content == "hello"
