"""Machine-readable claim policy helpers for RIPPLE-GWAS V1."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised in lean runtime environments.
    yaml = None

StatisticDirection = Literal["greater_is_more_extreme", "less_is_more_extreme", "two_sided"]

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "config" / "claim_policy.yaml"


def load_claim_policy(path: Path | str | None = None) -> dict[str, Any]:
    """Load the RIPPLE manuscript claim policy."""

    policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
    with policy_path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    policy = yaml.safe_load(text) if yaml is not None else _parse_simple_yaml(text)
    if not isinstance(policy, dict):
        raise ValueError(f"Claim policy must be a mapping: {policy_path}")
    return policy


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the limited YAML subset used by `claim_policy.yaml`.

    This fallback keeps policy loading available in lean runtimes that do not
    have PyYAML installed. It intentionally supports only nested mappings,
    scalar values, and scalar lists.
    """

    raw_lines = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        raw_lines.append((indent, line.strip()))

    def parse_scalar(value: str) -> Any:
        if value in {"true", "True"}:
            return True
        if value in {"false", "False"}:
            return False
        try:
            if any(ch in value for ch in (".", "e", "E")):
                return float(value)
            return int(value)
        except ValueError:
            return value.strip("\"'")

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(raw_lines):
            return {}, index
        is_list = raw_lines[index][0] == indent and raw_lines[index][1].startswith("- ")
        if is_list:
            values = []
            while index < len(raw_lines):
                line_indent, content = raw_lines[index]
                if line_indent != indent or not content.startswith("- "):
                    break
                values.append(parse_scalar(content[2:].strip()))
                index += 1
            return values, index

        values: dict[str, Any] = {}
        while index < len(raw_lines):
            line_indent, content = raw_lines[index]
            if line_indent != indent or content.startswith("- "):
                break
            if ":" not in content:
                raise ValueError(f"Unsupported policy YAML line: {content}")
            key, value = content.split(":", 1)
            key = key.strip()
            value = value.strip()
            index += 1
            if value:
                values[key] = parse_scalar(value)
            else:
                child, index = parse_block(index, indent + 2)
                values[key] = child
        return values, index

    parsed, final_index = parse_block(0, 0)
    if final_index != len(raw_lines):
        raise ValueError("Could not parse complete policy YAML file.")
    if not isinstance(parsed, dict):
        raise ValueError("Policy YAML root must be a mapping.")
    return parsed


def final_z_threshold(policy: dict[str, Any] | None = None) -> float:
    """Return the final-positive Z threshold from policy."""

    active = policy or load_claim_policy()
    return float(active["thresholds"]["final_z_threshold"])


def supportive_z_threshold(policy: dict[str, Any] | None = None) -> float:
    """Return the supportive Z threshold from policy."""

    active = policy or load_claim_policy()
    return float(active["thresholds"]["supportive_z_threshold"])


def classify_z_claim(z: object, policy: dict[str, Any] | None = None) -> str:
    """Classify a Z statistic using policy-defined final/supportive thresholds."""

    try:
        value = float(z)
    except (TypeError, ValueError):
        return "not_tested"
    if not np.isfinite(value):
        return "not_tested"
    active = policy or load_claim_policy()
    if value >= final_z_threshold(active):
        return "final_positive"
    if value >= supportive_z_threshold(active):
        return "supportive"
    return "negative"


def empirical_p_from_null(
    null_statistics: np.ndarray | list[float],
    observed: float,
    *,
    direction: StatisticDirection = "greater_is_more_extreme",
) -> float:
    """Compute empirical P with the RIPPLE V1 +1 correction."""

    null = np.asarray(null_statistics, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(float(observed)):
        return float("nan")
    obs = float(observed)
    if direction == "greater_is_more_extreme":
        exceedances = int(np.count_nonzero(null >= obs))
    elif direction == "less_is_more_extreme":
        exceedances = int(np.count_nonzero(null <= obs))
    elif direction == "two_sided":
        center = float(np.mean(null))
        exceedances = int(np.count_nonzero(np.abs(null - center) >= abs(obs - center)))
    else:
        raise ValueError(f"Unsupported statistic direction: {direction}")
    return float((1 + exceedances) / (1 + null.size))


def controlled_vocabulary(policy: dict[str, Any] | None, name: str) -> set[str]:
    """Return a named controlled vocabulary from the policy."""

    active = policy or load_claim_policy()
    values = active.get(name)
    if not isinstance(values, list):
        raise KeyError(f"Policy does not define controlled vocabulary {name!r}")
    return {str(value) for value in values}
