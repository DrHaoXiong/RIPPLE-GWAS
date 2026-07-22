"""Library, selection, provenance, and output contracts for RIPPLE-D V1.7."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd

from ripple.modules.v17_config import RippleDV17Config


LibraryRole = Literal["broad_discovery", "fixed_panel", "replication", "post_selection_followup"]
SelectionStage = Literal["none", "selected_from_same_trait", "selected_externally"]
LIBRARY_ROLES = frozenset(("broad_discovery", "fixed_panel", "replication", "post_selection_followup"))
SELECTION_STAGES = frozenset(("none", "selected_from_same_trait", "selected_externally"))


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 checksum of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object, name: str) -> str:
    value = str(value).strip()
    if not value:
        raise ValueError(f"{name} must not be empty")
    return value


@dataclass(frozen=True)
class SelectionProvenance:
    """Documents how a library entered the current analysis."""

    selection_stage: SelectionStage
    selected_from_scope_id: str | None = None
    selection_rule: str = "pre_specified"
    selection_source: str = "pre_registered"

    def __post_init__(self) -> None:
        if self.selection_stage not in SELECTION_STAGES:
            raise ValueError(f"unknown selection_stage: {self.selection_stage}")
        object.__setattr__(self, "selection_rule", _text(self.selection_rule, "selection_rule"))
        object.__setattr__(self, "selection_source", _text(self.selection_source, "selection_source"))
        if self.selection_stage == "none" and self.selected_from_scope_id is not None:
            raise ValueError("selection_stage='none' cannot name a parent scope")
        if self.selection_stage != "none" and not self.selected_from_scope_id:
            raise ValueError("selected library stages require selected_from_scope_id")


@dataclass(frozen=True)
class AnalysisScope:
    """The multiplicity family and claim authority of an analysis run."""

    correction_scope_id: str
    registered_library_fingerprint: str
    claim_level: Literal["manuscript_candidate", "diagnostic_only"] = "diagnostic_only"
    scope_complete: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "correction_scope_id", _text(self.correction_scope_id, "correction_scope_id")
        )
        fingerprint = _text(self.registered_library_fingerprint, "registered_library_fingerprint")
        if len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint.lower()):
            raise ValueError("registered_library_fingerprint must be a SHA-256 hex digest")
        object.__setattr__(self, "registered_library_fingerprint", fingerprint.lower())


@dataclass(frozen=True)
class LibraryManifest:
    """Immutable registration record for one tested library."""

    library_id: str
    role: LibraryRole
    selection: SelectionProvenance
    scope: AnalysisScope
    provenance: Mapping[str, str]
    file_checksums: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "library_id", _text(self.library_id, "library_id"))
        if self.role not in LIBRARY_ROLES:
            raise ValueError(f"unknown library role: {self.role}")
        if self.role == "broad_discovery" and self.selection.selection_stage != "none":
            raise ValueError("broad_discovery libraries must use selection_stage='none'")
        if self.role in {"fixed_panel", "replication", "post_selection_followup"} and self.selection.selection_stage == "none":
            raise ValueError("selected library roles require explicit selection provenance")
        provenance = {
            str(key): _text(value, f"provenance[{key}]")
            for key, value in self.provenance.items()
        }
        required = {"library_source", "library_version", "created_by"}
        missing = sorted(required - set(provenance))
        if missing:
            raise ValueError(f"provenance is missing required fields: {missing}")
        object.__setattr__(self, "provenance", dict(sorted(provenance.items())))
        checksums = {str(key): str(value).lower() for key, value in self.file_checksums.items()}
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in checksums.values()
        ):
            raise ValueError("file_checksums values must be SHA-256 hex digests")
        object.__setattr__(self, "file_checksums", dict(sorted(checksums.items())))

    def canonical_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def validate_files(self, *, base_dir: str | Path = ".") -> None:
        root = Path(base_dir)
        for filename, expected in self.file_checksums.items():
            actual = sha256_file(root / filename)
            if actual != expected:
                raise ValueError(f"checksum mismatch for {filename}: expected {expected}, got {actual}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LibraryManifest":
        values = dict(data)
        if isinstance(values.get("selection"), Mapping):
            values["selection"] = SelectionProvenance(**values["selection"])
        if isinstance(values.get("scope"), Mapping):
            values["scope"] = AnalysisScope(**values["scope"])
        return cls(**values)


def load_library_manifest(path: str | Path) -> LibraryManifest:
    """Load exactly one library manifest record from JSON or TSV."""

    manifests = load_library_manifests(path)
    if len(manifests) != 1:
        raise ValueError(f"expected one library manifest, found {len(manifests)}")
    return manifests[0]


def load_library_manifests(path: str | Path) -> tuple[LibraryManifest, ...]:
    """Load JSON object/list or TSV manifest records and validate their contracts."""

    path = Path(path)
    if path.suffix.lower() == ".tsv":
        records = pd.read_csv(path, sep="\t", dtype=str).fillna("").to_dict("records")
        parsed = []
        for row in records:
            parsed.append(
                {
                    "library_id": row["library_id"],
                    "role": row["role"],
                    "selection": {
                        "selection_stage": row["selection_stage"],
                        "selected_from_scope_id": row.get("selected_from_scope_id") or None,
                        "selection_rule": row.get("selection_rule") or "pre_specified",
                        "selection_source": row.get("selection_source") or "pre_registered",
                    },
                    "scope": {
                        "correction_scope_id": row["correction_scope_id"],
                        "registered_library_fingerprint": row["registered_library_fingerprint"],
                        "claim_level": row.get("claim_level") or "diagnostic_only",
                        "scope_complete": str(row.get("scope_complete", "true")).lower()
                        in {"true", "1", "yes"},
                    },
                    "provenance": {
                        "library_source": row["library_source"],
                        "library_version": row["library_version"],
                        "created_by": row["created_by"],
                    },
                }
            )
        return tuple(LibraryManifest.from_dict(record) for record in parsed)
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("libraries", []) if isinstance(data, Mapping) else data
    if not isinstance(records, list):
        raise ValueError("JSON manifest must be a list or an object containing 'libraries'")
    return tuple(LibraryManifest.from_dict(record) for record in records)


@dataclass(frozen=True)
class OutputTableSchema:
    name: str
    required_columns: tuple[str, ...]
    q_column: str | None = None

    def validate(self, table: pd.DataFrame, *, manifest: LibraryManifest | None = None) -> None:
        missing = sorted(set(self.required_columns) - set(table.columns))
        if missing:
            raise ValueError(f"{self.name} is missing required columns: {missing}")
        if table.columns.duplicated().any():
            raise ValueError(f"{self.name} contains duplicate columns")
        if self.q_column and self.q_column in table:
            q = pd.to_numeric(table[self.q_column], errors="coerce")
            finite = q.notna()
            if ((q[finite] < 0) | (q[finite] > 1)).any():
                raise ValueError(f"{self.name}.{self.q_column} must be in [0, 1]")
            if "test_status" in table and q.loc[table["test_status"].ne("tested")].notna().any():
                raise ValueError("q values are valid only for tested rows")
        if manifest is not None:
            for column, expected in (
                ("correction_scope_id", manifest.scope.correction_scope_id),
                ("library_fingerprint", manifest.scope.registered_library_fingerprint),
            ):
                if column in table and not table[column].dropna().astype(str).eq(expected).all():
                    raise ValueError(f"{self.name}.{column} does not match the registered manifest")
            if manifest.role == "post_selection_followup" and self.q_column and self.q_column in table:
                raise ValueError("post_selection_followup q values cannot support a discovery claim")


V17_OUTPUT_TABLE_SCHEMAS = {
    "module_tests": OutputTableSchema(
        "module_tests",
        (
            "module_name", "test_status", "trait", "cohort", "library_id",
            "parent_library_id", "library_role", "selection_stage", "selection_trait",
            "multiplicity_scope", "multiplicity_m", "q_value_valid_for_discovery",
            "correction_scope_id", "n_null",
            "v17_adaptive_omnibus_empirical_p", "v17_adaptive_omnibus_bh_q",
            "library_fingerprint", "analysis_scope_fingerprint", "config_hash",
            "v17_nomination_state", "v17_nomination_downgrade_reason",
        ),
        "v17_adaptive_omnibus_bh_q",
    ),
    "external_locus_audit": OutputTableSchema("external_locus_audit", ("external_locus_audit_pass",)),
    "summary": OutputTableSchema("summary", ("library_role", "correction_scope_id", "n_null", "library_fingerprint", "analysis_scope_fingerprint")),
}


def validate_output_table(name: str, table: pd.DataFrame, *, manifest: LibraryManifest | None = None) -> None:
    try:
        schema = V17_OUTPUT_TABLE_SCHEMAS[name]
    except KeyError as exc:
        raise ValueError(f"unknown V1.7 output table schema: {name}") from exc
    schema.validate(table, manifest=manifest)


def write_analysis_manifest(
    path: str | Path,
    *,
    config: RippleDV17Config,
    libraries: Sequence[LibraryManifest],
    inputs: Mapping[str, str | Path] | None = None,
    outputs: Mapping[str, str | Path] | None = None,
    extra_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a checksum-bound analysis manifest and return its JSON payload."""

    payload = {
        "schema_version": "ripple_v17_analysis_manifest_v1",
        "config": config.canonical_dict(),
        "config_hash": config.canonical_hash,
        "libraries": [item.canonical_dict() for item in libraries],
        "input_checksums": {
            str(name): sha256_file(value) for name, value in sorted((inputs or {}).items())
        },
        "output_checksums": {
            str(name): sha256_file(value) for name, value in sorted((outputs or {}).items())
        },
        "provenance": dict(
            sorted((str(key), value) for key, value in (extra_provenance or {}).items())
        ),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
