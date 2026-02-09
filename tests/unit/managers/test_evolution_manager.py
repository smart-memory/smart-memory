"""Unit tests for EvolutionManager."""

from unittest.mock import MagicMock, call

import pytest

pytestmark = pytest.mark.unit

from smartmemory.managers.evolution import EvolutionManager


@pytest.fixture
def mock_evolution_orchestrator():
    orch = MagicMock()
    orch.run_evolution_cycle.return_value = {"promoted": 3, "decayed": 1}
    orch.commit_working_to_episodic.return_value = ["item_1", "item_2"]
    orch.commit_working_to_procedural.return_value = ["item_3"]
    return orch


@pytest.fixture
def mock_clustering():
    cl = MagicMock()
    cl.run.return_value = {"merged": 2, "clusters": 5}
    return cl


@pytest.fixture
def manager(mock_evolution_orchestrator, mock_clustering):
    return EvolutionManager(mock_evolution_orchestrator, mock_clustering)


class TestRunEvolutionCycle:
    def test_delegates_to_orchestrator(self, manager, mock_evolution_orchestrator):
        result = manager.run_evolution_cycle()
        mock_evolution_orchestrator.run_evolution_cycle.assert_called_once()
        assert result == {"promoted": 3, "decayed": 1}

    def test_propagates_orchestrator_exception(self, manager, mock_evolution_orchestrator):
        mock_evolution_orchestrator.run_evolution_cycle.side_effect = RuntimeError("DB down")
        with pytest.raises(RuntimeError, match="DB down"):
            manager.run_evolution_cycle()


class TestCommitWorkingToEpisodic:
    def test_default_remove_from_source(self, manager, mock_evolution_orchestrator):
        result = manager.commit_working_to_episodic()
        mock_evolution_orchestrator.commit_working_to_episodic.assert_called_once_with(True)
        assert result == ["item_1", "item_2"]

    def test_keep_source(self, manager, mock_evolution_orchestrator):
        manager.commit_working_to_episodic(remove_from_source=False)
        mock_evolution_orchestrator.commit_working_to_episodic.assert_called_once_with(False)

    def test_returns_promoted_ids(self, manager):
        result = manager.commit_working_to_episodic()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_empty_result(self, manager, mock_evolution_orchestrator):
        mock_evolution_orchestrator.commit_working_to_episodic.return_value = []
        result = manager.commit_working_to_episodic()
        assert result == []


class TestCommitWorkingToProcedural:
    def test_default_remove_from_source(self, manager, mock_evolution_orchestrator):
        result = manager.commit_working_to_procedural()
        mock_evolution_orchestrator.commit_working_to_procedural.assert_called_once_with(True)
        assert result == ["item_3"]

    def test_keep_source(self, manager, mock_evolution_orchestrator):
        manager.commit_working_to_procedural(remove_from_source=False)
        mock_evolution_orchestrator.commit_working_to_procedural.assert_called_once_with(False)


class TestRunClustering:
    def test_delegates_to_clustering(self, manager, mock_clustering):
        result = manager.run_clustering()
        mock_clustering.run.assert_called_once()
        assert result == {"merged": 2, "clusters": 5}

    def test_propagates_clustering_exception(self, manager, mock_clustering):
        mock_clustering.run.side_effect = ValueError("No data")
        with pytest.raises(ValueError, match="No data"):
            manager.run_clustering()
