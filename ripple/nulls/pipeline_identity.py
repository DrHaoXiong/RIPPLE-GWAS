"""Observed/null pipeline identity validation."""

from __future__ import annotations

from ripple.contracts import PipelineStageManifest


def validate_pipeline_identity(
    observed: PipelineStageManifest,
    null: PipelineStageManifest,
) -> None:
    """Raise if observed and null runs did not execute identical stages."""

    if observed.stages != null.stages:
        raise ValueError(
            "Observed/null pipeline mismatch: "
            f"observed={observed.stages}, null={null.stages}"
        )
