"""Unit tests for interfaces module — ScopeProvider ABC and DefaultScopeProvider compliance."""

from abc import ABC

import pytest

pytestmark = pytest.mark.unit

from smartmemory.interfaces import ScopeProvider
from smartmemory.scope_provider import DefaultScopeProvider


class TestScopeProviderABC:
    def test_is_abstract(self):
        assert issubclass(ScopeProvider, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ScopeProvider()

    def test_required_abstract_methods(self):
        abstract_methods = ScopeProvider.__abstractmethods__
        assert "get_isolation_filters" in abstract_methods
        assert "get_write_context" in abstract_methods
        assert "get_global_search_filters" in abstract_methods
        assert "get_user_isolation_key" in abstract_methods

    def test_incomplete_subclass_raises(self):
        class PartialProvider(ScopeProvider):
            def get_isolation_filters(self):
                return {}

        with pytest.raises(TypeError):
            PartialProvider()

    def test_complete_subclass_instantiates(self):
        class FullProvider(ScopeProvider):
            def get_isolation_filters(self):
                return {"tenant_id": "t1"}

            def get_write_context(self):
                return {"tenant_id": "t1"}

            def get_global_search_filters(self):
                return {"tenant_id": "t1"}

            def get_user_isolation_key(self):
                return "user_id"

        provider = FullProvider()
        assert provider.get_isolation_filters() == {"tenant_id": "t1"}
        assert provider.get_user_isolation_key() == "user_id"


class TestDefaultScopeProviderCompliance:
    """Verify DefaultScopeProvider correctly implements ScopeProvider interface."""

    def test_is_subclass_of_scope_provider(self):
        assert issubclass(DefaultScopeProvider, ScopeProvider)

    def test_default_no_scope(self):
        provider = DefaultScopeProvider()
        assert provider.get_isolation_filters() == {}
        assert provider.get_write_context() == {}
        assert provider.get_global_search_filters() == {}
        assert provider.get_user_isolation_key() == "user_id"

    def test_tenant_only(self):
        provider = DefaultScopeProvider(tenant_id="t1")
        filters = provider.get_isolation_filters()
        assert filters == {"tenant_id": "t1"}

    def test_full_scope(self):
        provider = DefaultScopeProvider(
            tenant_id="t1", workspace_id="w1", user_id="u1", team_id="team1"
        )
        filters = provider.get_isolation_filters()
        assert filters == {"tenant_id": "t1", "workspace_id": "w1", "user_id": "u1"}

    def test_write_context_includes_team(self):
        provider = DefaultScopeProvider(
            tenant_id="t1", workspace_id="w1", user_id="u1", team_id="team1"
        )
        ctx = provider.get_write_context()
        assert ctx["team_id"] == "team1"
        assert ctx["tenant_id"] == "t1"

    def test_write_context_no_team(self):
        provider = DefaultScopeProvider(tenant_id="t1")
        ctx = provider.get_write_context()
        assert "team_id" not in ctx

    def test_global_search_excludes_user(self):
        provider = DefaultScopeProvider(
            tenant_id="t1", workspace_id="w1", user_id="u1"
        )
        filters = provider.get_global_search_filters()
        assert "user_id" not in filters
        assert filters["tenant_id"] == "t1"
        assert filters["workspace_id"] == "w1"

    def test_global_search_no_scope(self):
        provider = DefaultScopeProvider()
        assert provider.get_global_search_filters() == {}
