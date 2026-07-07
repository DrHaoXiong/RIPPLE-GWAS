"""Observed/null pipeline orchestration skeleton."""

from __future__ import annotations

from typing import Literal

from ripple.contracts import PipelineResult, PipelineStageManifest, RippleConfig
from ripple.defaults import REQUIRED_PIPELINE_STAGES


def run_pipeline(config: RippleConfig, mode: Literal["observed", "null"]) -> PipelineResult:
    """Run one observed or null pipeline.

    The implementation must execute the same ordered stages for observed and null data.
    """

    manifest = PipelineStageManifest(mode=mode, stages=REQUIRED_PIPELINE_STAGES)
    return PipelineResult(config=config, manifest=manifest)
