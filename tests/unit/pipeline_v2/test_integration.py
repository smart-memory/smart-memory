"""Integration test -- run the full pipeline with mocked dependencies."""

from unittest.mock import MagicMock, patch
from dataclasses import replace

from smartmemory.pipeline.config import PipelineConfig
from smartmemory.pipeline.runner import PipelineRunner
from smartmemory.pipeline.state import PipelineState
from smartmemory.pipeline.stages.classify import ClassifyStage
from smartmemory.pipeline.stages.coreference import CoreferenceStageCommand
from smartmemory.pipeline.stages.extract import ExtractStage
from smartmemory.pipeline.stages.store import StoreStage
from smartmemory.pipeline.stages.link import LinkStage
from smartmemory.pipeline.stages.enrich import EnrichStage
from smartmemory.pipeline.stages.ground import GroundStage
from smartmemory.pipeline.stages.evolve import EvolveStage


# ------------------------------------------------------------------ #
# Mock helpers
# ------------------------------------------------------------------ #


def _mock_ingestion_flow():
    flow = MagicMock()
    flow.classify_item.return_value = ["semantic"]
    return flow


def _mock_extraction_pipeline():
    pipeline = MagicMock()
    pipeline.extract_semantics.return_value = {
        "entities": [{"name": "Claude", "type": "Technology"}],
        "relations": [],
    }
    return pipeline


def _mock_memory():
    memory = MagicMock()
    memory._crud.add.return_value = {
        "memory_node_id": "test_item_123",
        "entity_node_ids": ["e1"],
    }
    memory._evolution = MagicMock()
    memory._clustering = MagicMock()
    memory._graph = MagicMock()
    memory._grounding = MagicMock()
    return memory


def _mock_linking():
    return MagicMock()


def _mock_enrichment_pipeline():
    pipeline = MagicMock()
    pipeline.run_enrichment.return_value = {"basic": "done"}
    return pipeline


def _build_all_stages():
    """Create the full 8-stage pipeline with mocked dependencies."""
    flow = _mock_ingestion_flow()
    extraction = _mock_extraction_pipeline()
    memory = _mock_memory()
    linking = _mock_linking()
    enrichment = _mock_enrichment_pipeline()

    return [
        ClassifyStage(flow),
        CoreferenceStageCommand(),
        ExtractStage(extraction),
        StoreStage(memory),
        LinkStage(linking),
        EnrichStage(enrichment),
        GroundStage(memory),
        EvolveStage(memory),
    ]


STAGE_NAMES = [
    "classify",
    "coreference",
    "extract",
    "store",
    "link",
    "enrich",
    "ground",
    "evolve",
]


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: (
        # Handle both (state, config) and accidental calls
        replace(
            self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
            resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
        )
    ),
)
def test_full_pipeline_runs_all_stages(_mock_coref):
    """All 8 stages execute and are recorded in stage_history."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.default(workspace_id="test")

    state = runner.run("Claude is an AI assistant.", config)

    assert len(state.stage_history) == 8
    assert state.stage_history == STAGE_NAMES
    assert state.completed_at is not None
    assert state.item_id == "test_item_123"


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
    ),
)
def test_pipeline_preview_mode(_mock_coref):
    """Preview mode returns preview_item and skips evolution/clustering."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.preview(workspace_id="test")

    state = runner.run("Preview test content.", config)

    assert state.item_id == "preview_item"
    # Evolution should not have been called in preview mode
    memory = stages[3]._memory  # StoreStage holds the memory mock
    memory._evolution.run_evolution_cycle.assert_not_called()
    memory._clustering.run.assert_not_called()


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
    ),
)
def test_pipeline_run_to_extract(_mock_coref):
    """run_to('extract') only executes classify, coreference, extract."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.default(workspace_id="test")

    state = runner.run_to("Claude is an AI.", config, stop_after="extract")

    assert state.stage_history == ["classify", "coreference", "extract"]
    assert "store" not in state.stage_history
    assert "link" not in state.stage_history
    assert "enrich" not in state.stage_history
    assert "ground" not in state.stage_history
    assert "evolve" not in state.stage_history


@patch(
    "smartmemory.pipeline.stages.coreference.CoreferenceStageCommand.execute",
    side_effect=lambda self_or_state, config_or_none=None, **kw: replace(
        self_or_state if isinstance(self_or_state, PipelineState) else config_or_none,
        resolved_text=(self_or_state if isinstance(self_or_state, PipelineState) else config_or_none).text,
    ),
)
def test_pipeline_run_from_checkpoint(_mock_coref):
    """run_from() resumes from a checkpoint and completes remaining stages."""
    stages = _build_all_stages()
    runner = PipelineRunner(stages)
    config = PipelineConfig.default(workspace_id="test")

    # Phase 1: run up to extract
    partial_state = runner.run_to("Claude is an AI.", config, stop_after="extract")
    assert partial_state.stage_history == ["classify", "coreference", "extract"]

    # Phase 2: resume from checkpoint
    final_state = runner.run_from(partial_state, config)

    assert final_state.stage_history == [
        "classify",
        "coreference",
        "extract",
        "store",
        "link",
        "enrich",
        "ground",
        "evolve",
    ]
    assert final_state.completed_at is not None
    assert final_state.item_id == "test_item_123"
