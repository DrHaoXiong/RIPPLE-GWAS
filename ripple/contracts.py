"""Shared data contracts for the RIPPLE V1 private prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ripple.defaults import (
    DEFAULT_LD_SHRINKAGE,
    PRIMARY_RESIDUALIZATION_COVARIATES,
    RANK_FRACTION_GRID,
)


@dataclass(frozen=True)
class RippleConfig:
    """Configuration shared by observed and null pipeline runs."""

    trait: str
    graph_id: str
    ld_shrinkage: float = DEFAULT_LD_SHRINKAGE
    rank_fraction_grid: tuple[float, ...] = RANK_FRACTION_GRID
    residualization_covariates: tuple[str, ...] = PRIMARY_RESIDUALIZATION_COVARIATES
    include_degree_in_primary_residualization: bool = False
    p_clip_epsilon: float = 1e-15


@dataclass(frozen=True)
class PipelineStageManifest:
    """Records pipeline stage execution for identity checks."""

    mode: Literal["observed", "null"]
    stages: tuple[str, ...]
    notes: tuple[str, ...] = ()


@dataclass
class PipelineResult:
    """Container for one observed or null run."""

    config: RippleConfig
    manifest: PipelineStageManifest
    tables: dict[str, object] = field(default_factory=dict)
