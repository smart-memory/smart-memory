"""Unit tests for utils.context — user context management via contextvars."""

import pytest

pytestmark = pytest.mark.unit

from smartmemory.utils.context import set_user_id, get_user_id, current_user_id


@pytest.fixture(autouse=True)
def _reset_context():
    """Reset context before and after each test."""
    set_user_id(None)
    yield
    set_user_id(None)


class TestUserIdContext:
    def test_default_is_none(self):
        assert get_user_id() is None

    def test_set_and_get(self):
        set_user_id("user_123")
        assert get_user_id() == "user_123"

    def test_clear(self):
        set_user_id("user_123")
        set_user_id(None)
        assert get_user_id() is None

    def test_current_user_id_alias(self):
        set_user_id("user_456")
        assert current_user_id() == "user_456"

    def test_overwrite(self):
        set_user_id("first")
        set_user_id("second")
        assert get_user_id() == "second"
