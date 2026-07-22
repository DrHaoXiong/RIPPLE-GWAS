"""Immutable, serializable configuration for the RIPPLE-D V1.7 analysis."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ripple.modules.adaptive import AdaptiveLocusConfig, V17_COMPONENTS
from ripple.modules.distributed import RippleDConfig


def _required_text(value: str, name: str) -> str:
    value = str(value).strip()
    if not value or value.lower() in {"unknown", "unspecified"}:
        raise ValueError(f"{name} must be a non-empty, specified string")
    return value


@dataclass(frozen=True)
class ExternalLocusMetadata:
    """Identity and provenance of the externally supplied locus definition."""

    locus_id_column: str
    locus_definition_name: str
    locus_source: str
    locus_source_version: str
    genome_build: str
    ancestry: str
    construction_script: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            object.__setattr__(self, name, _required_text(value, name))


@dataclass(frozen=True)
class MatchingConfig:
    """Frozen locus-matched null sampling design."""

    locus_collapse: str = "max"
    degree_bins: int = 5
    property_bins: int = 2
    annotation_matching_enabled: bool = False
    require_gene_count_match: bool = True
    null_gene_subset_sampling: bool = True

    def __post_init__(self) -> None:
        if self.locus_collapse != "max":
            raise ValueError("V1.7 requires locus_collapse='max'")
        if self.degree_bins < 1 or self.property_bins < 1:
            raise ValueError("degree_bins and property_bins must be positive")


@dataclass(frozen=True)
class NullPolicy:
    """Quality gates applied to the matched-locus null draws."""

    n_null: int = 10_000
    exact_match_rate_min: float = 0.80
    global_fallback_rate_max: float = 0.05
    reuse_fallback_rate_max: float = 0.0
    within_locus_replacement_rate_max: float = 0.0

    def __post_init__(self) -> None:
        if self.n_null < 1:
            raise ValueError("n_null must be positive")
        for name, value in asdict(self).items():
            if name != "n_null" and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class QPolicy:
    """Multiplicity and empirical-resolution policy for one registered scope."""

    q_max: float = 0.10
    correction_method: str = "benjamini_hochberg"
    require_scope_complete: bool = True
    require_empirical_resolution: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.q_max <= 1.0:
            raise ValueError("q_max must be in (0, 1]")
        if self.correction_method != "benjamini_hochberg":
            raise ValueError("V1.7 requires benjamini_hochberg correction")


@dataclass(frozen=True)
class RippleDV17Config:
    """Complete immutable V1.7 configuration with stable JSON identity."""

    external_locus: ExternalLocusMetadata
    matching: MatchingConfig = MatchingConfig()
    null_policy: NullPolicy = NullPolicy()
    q_policy: QPolicy = QPolicy()
    components: tuple[str, ...] = V17_COMPONENTS
    topk_fractions: tuple[float, ...] = (0.10, 0.20, 0.50, 1.0)
    topk_min_loci: int = 3
    min_present_genes: int = 5
    dispersion_effective_loci_min: float = 5.0
    leave_top1_empirical_p_max: float = 0.025
    leave_top1_supportive_p_max: float = 0.10
    score_cap: float = 3.0

    def __post_init__(self) -> None:
        components = tuple(str(value) for value in self.components)
        if components != tuple(V17_COMPONENTS):
            raise ValueError("components must equal the frozen V1.7 component family in order")
        object.__setattr__(self, "components", components)
        fractions = tuple(float(value) for value in self.topk_fractions)
        if not fractions or any(not 0.0 < value <= 1.0 for value in fractions):
            raise ValueError("topk_fractions must contain values in (0, 1]")
        if tuple(sorted(set(fractions))) != fractions or fractions[-1] != 1.0:
            raise ValueError("topk_fractions must be unique, sorted, and include 1.0")
        object.__setattr__(self, "topk_fractions", fractions)
        if self.topk_min_loci < 1 or self.min_present_genes < 1:
            raise ValueError("topk_min_loci and min_present_genes must be positive")
        if self.dispersion_effective_loci_min <= 0 or self.score_cap <= 0:
            raise ValueError("dispersion_effective_loci_min and score_cap must be positive")
        if not 0 < self.leave_top1_empirical_p_max <= self.leave_top1_supportive_p_max <= 1:
            raise ValueError("leave-top-1 thresholds must be ordered values in (0, 1]")

    def canonical_dict(self) -> dict[str, Any]:
        """Return a JSON-native representation with deterministic key ordering."""

        return json.loads(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def canonical_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_adaptive_locus_config(self) -> AdaptiveLocusConfig:
        """Adapt the public V1.7 contract to the existing analysis implementation."""

        external = self.external_locus
        matching = self.matching
        nulls = self.null_policy
        ripple_d = RippleDConfig(
            score_cap=self.score_cap,
            locus_collapse=matching.locus_collapse,
            degree_bins=matching.degree_bins,
            property_bins=matching.property_bins,
            annotation_matching_enabled=matching.annotation_matching_enabled,
            locus_id_column=external.locus_id_column,
            locus_definition_name=external.locus_definition_name,
            require_gene_count_match=matching.require_gene_count_match,
            null_gene_subset_sampling=matching.null_gene_subset_sampling,
        )
        return AdaptiveLocusConfig(
            ripple_d=ripple_d,
            topk_fractions=self.topk_fractions,
            topk_min_loci=self.topk_min_loci,
            min_present_genes=self.min_present_genes,
            q_max=self.q_policy.q_max,
            dispersion_effective_loci_min=self.dispersion_effective_loci_min,
            leave_top1_empirical_p_max=self.leave_top1_empirical_p_max,
            leave_top1_supportive_p_max=self.leave_top1_supportive_p_max,
            null_exact_match_rate_min=nulls.exact_match_rate_min,
            null_global_fallback_rate_max=nulls.global_fallback_rate_max,
            null_reuse_fallback_rate_max=nulls.reuse_fallback_rate_max,
            within_locus_replacement_rate_max=nulls.within_locus_replacement_rate_max,
        )

    def validate(self) -> None:
        """Validate this instance; useful at an explicit configuration boundary."""

        self.__post_init__()

    def dump_json(self, path: str | Path, *, indent: int = 2) -> None:
        Path(path).write_text(
            json.dumps(self.canonical_dict(), indent=indent, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    dump = dump_json

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RippleDV17Config":
        values = dict(data)
        for name, constructor in (
            ("external_locus", ExternalLocusMetadata),
            ("matching", MatchingConfig),
            ("null_policy", NullPolicy),
            ("q_policy", QPolicy),
        ):
            value = values.get(name)
            if isinstance(value, Mapping):
                values[name] = constructor(**value)
        if "components" in values:
            values["components"] = tuple(values["components"])
        if "topk_fractions" in values:
            values["topk_fractions"] = tuple(values["topk_fractions"])
        return cls(**values)

    @classmethod
    def load_json(cls, path: str | Path) -> "RippleDV17Config":
        with Path(path).open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, Mapping):
            raise ValueError("V1.7 config JSON must contain an object")
        return cls.from_dict(data)

    load = load_json
