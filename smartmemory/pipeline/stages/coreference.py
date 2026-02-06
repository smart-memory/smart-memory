"""Coreference stage — wraps CoreferenceStage.run()."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from smartmemory.pipeline.state import PipelineState

if TYPE_CHECKING:
    from smartmemory.pipeline.config import PipelineConfig

logger = logging.getLogger(__name__)


class CoreferenceStageCommand:
    """Resolve pronouns and vague references to explicit entity names."""

    @property
    def name(self) -> str:
        return "coreference"

    def execute(self, state: PipelineState, config: PipelineConfig) -> PipelineState:
        coref_cfg = config.coreference
        if not coref_cfg.enabled:
            return replace(state, resolved_text=state.text)

        text = state.text
        if not text or len(text) < coref_cfg.min_text_length:
            return replace(state, resolved_text=text)

        try:
            from smartmemory.memory.pipeline.stages.coreference import CoreferenceStage as LegacyCoref
            from smartmemory.memory.pipeline.config import CoreferenceConfig as LegacyCorefConfig

            coref_stage = LegacyCoref(min_text_length=coref_cfg.min_text_length)
            legacy_conf = LegacyCorefConfig(
                enabled=coref_cfg.enabled,
                resolver=coref_cfg.resolver,
                device=coref_cfg.device,
                min_text_length=coref_cfg.min_text_length,
            )
            result = coref_stage.run(text, config=legacy_conf)

            resolved = result.resolved_text if not result.skipped else text
            ctx = dict(state._context)
            ctx["coreference_result"] = {
                "skipped": result.skipped,
                "skip_reason": result.skip_reason,
                "chains": result.chains,
                "replacements_made": result.replacements_made,
            }
            return replace(state, resolved_text=resolved, _context=ctx)

        except Exception as e:
            logger.warning("Coreference resolution failed: %s", e)
            return replace(state, resolved_text=text)

    def undo(self, state: PipelineState) -> PipelineState:
        return replace(state, resolved_text=None)
