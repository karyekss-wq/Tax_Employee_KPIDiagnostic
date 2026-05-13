from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

from delta_analysis import build_simulation_deltas
from simulation import run_simulation


BASE_DIR = Path(__file__).resolve().parent
SCENARIO_DIR = BASE_DIR / "scenarios"

OVERRIDE_SECTION_KEYS = [
    "class_expected_hours_overrides",
    "class_weight_overrides",
    "adjustment_multiplier_overrides",
]

SCENARIO_KEYS = [
    "scenario_id",
    "scenario_name",
    "created_at",
    "updated_at",
    "overrides",
]

SAFE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _scenario_dir(storage_dir: Path | str | None = None) -> Path:
    return Path(storage_dir) if storage_dir is not None else SCENARIO_DIR


def make_scenario_id(scenario_name: str) -> str:
    """
    Build a deterministic filesystem-safe scenario id from a display name.
    """
    if not isinstance(scenario_name, str) or scenario_name.strip() == "":
        raise ValueError("scenario_name cannot be blank.")
    scenario_id = re.sub(r"[^a-z0-9]+", "_", scenario_name.strip().lower())
    scenario_id = re.sub(r"_+", "_", scenario_id).strip("_")
    if scenario_id == "":
        raise ValueError("scenario_name collapses to an empty scenario_id.")
    return scenario_id


def validate_scenario_id(scenario_id: str) -> str:
    if not isinstance(scenario_id, str) or scenario_id.strip() == "":
        raise ValueError("scenario_id cannot be blank.")
    if scenario_id != scenario_id.strip():
        raise ValueError("scenario_id must not contain leading or trailing whitespace.")
    if not SAFE_ID_PATTERN.fullmatch(scenario_id):
        raise ValueError("scenario_id must contain only lowercase letters, numbers, and underscores.")
    return scenario_id


def _scenario_path(scenario_id: str, storage_dir: Path | str | None = None) -> Path:
    safe_id = validate_scenario_id(scenario_id)
    return _scenario_dir(storage_dir) / f"{safe_id}.json"


def _validate_override_value(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def validate_overrides(overrides: dict[str, Any]) -> dict[str, dict[str, float]]:
    if not isinstance(overrides, dict):
        raise ValueError("overrides must be a dict.")

    actual_keys = set(overrides.keys())
    expected_keys = set(OVERRIDE_SECTION_KEYS)
    unknown = sorted(actual_keys - expected_keys)
    missing = sorted(expected_keys - actual_keys)
    if unknown:
        raise ValueError(f"Unknown override section(s): {unknown}")
    if missing:
        raise ValueError(f"Missing override section(s): {missing}")

    validated: dict[str, dict[str, float]] = {}
    for section in OVERRIDE_SECTION_KEYS:
        section_value = overrides[section]
        if not isinstance(section_value, dict):
            raise ValueError(f"{section} must be a dict.")
        validated[section] = {}
        for config_key, raw_value in section_value.items():
            if not isinstance(config_key, str) or config_key == "":
                raise ValueError(f"{section} contains a blank or non-string override key.")
            validated[section][config_key] = _validate_override_value(
                raw_value, f"{section}.{config_key}"
            )
    return validated


def build_override_state(
    *,
    class_expected_hours_overrides: dict[str, Any] | None = None,
    class_weight_overrides: dict[str, Any] | None = None,
    adjustment_multiplier_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Build the explicit canonical override state. Empty sections are intentional.
    """
    return validate_overrides(
        {
            "class_expected_hours_overrides": deepcopy(class_expected_hours_overrides or {}),
            "class_weight_overrides": deepcopy(class_weight_overrides or {}),
            "adjustment_multiplier_overrides": deepcopy(adjustment_multiplier_overrides or {}),
        }
    )


def validate_scenario_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("scenario record must be a dict.")

    actual_keys = set(record.keys())
    expected_keys = set(SCENARIO_KEYS)
    unknown = sorted(actual_keys - expected_keys)
    missing = sorted(expected_keys - actual_keys)
    if unknown:
        raise ValueError(f"Unknown scenario field(s): {unknown}")
    if missing:
        raise ValueError(f"Missing scenario field(s): {missing}")

    scenario_name = record["scenario_name"]
    if not isinstance(scenario_name, str) or scenario_name.strip() == "":
        raise ValueError("scenario_name cannot be blank.")

    validated = {
        "scenario_id": validate_scenario_id(record["scenario_id"]),
        "scenario_name": scenario_name,
        "created_at": _validate_timestamp(record["created_at"], "created_at"),
        "updated_at": _validate_timestamp(record["updated_at"], "updated_at"),
        "overrides": validate_overrides(record["overrides"]),
    }
    return validated


def _validate_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty ISO timestamp string.")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO timestamp string.") from exc
    return value


def build_scenario_record(
    *,
    scenario_name: str,
    class_expected_hours_overrides: dict[str, Any] | None = None,
    class_weight_overrides: dict[str, Any] | None = None,
    adjustment_multiplier_overrides: dict[str, Any] | None = None,
    scenario_id: str | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    if not isinstance(scenario_name, str) or scenario_name.strip() == "":
        raise ValueError("scenario_name cannot be blank.")

    resolved_id = validate_scenario_id(scenario_id) if scenario_id else make_scenario_id(scenario_name)
    timestamp = _now_iso()
    record = {
        "scenario_id": resolved_id,
        "scenario_name": scenario_name,
        "created_at": created_at or timestamp,
        "updated_at": updated_at or timestamp,
        "overrides": build_override_state(
            class_expected_hours_overrides=class_expected_hours_overrides,
            class_weight_overrides=class_weight_overrides,
            adjustment_multiplier_overrides=adjustment_multiplier_overrides,
        ),
    }
    return validate_scenario_record(record)


def save_scenario(
    *,
    scenario_name: str,
    class_expected_hours_overrides: dict[str, Any] | None = None,
    class_weight_overrides: dict[str, Any] | None = None,
    adjustment_multiplier_overrides: dict[str, Any] | None = None,
    scenario_id: str | None = None,
    overwrite: bool = False,
    storage_dir: Path | str | None = None,
) -> dict[str, Any]:
    record = build_scenario_record(
        scenario_name=scenario_name,
        class_expected_hours_overrides=class_expected_hours_overrides,
        class_weight_overrides=class_weight_overrides,
        adjustment_multiplier_overrides=adjustment_multiplier_overrides,
        scenario_id=scenario_id,
    )
    path = _scenario_path(record["scenario_id"], storage_dir)
    if path.exists() and not overwrite:
        raise ValueError(f"Scenario '{record['scenario_id']}' already exists.")

    if path.exists() and overwrite:
        existing = load_scenario(record["scenario_id"], storage_dir=storage_dir)
        record["created_at"] = existing["created_at"]
        record["updated_at"] = _now_iso()
        record = validate_scenario_record(record)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def load_scenario(scenario_id: str, storage_dir: Path | str | None = None) -> dict[str, Any]:
    path = _scenario_path(scenario_id, storage_dir)
    if not path.exists():
        raise FileNotFoundError(f"Scenario '{scenario_id}' does not exist.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Scenario '{scenario_id}' contains malformed JSON.") from exc
    return validate_scenario_record(raw)


def list_scenarios(storage_dir: Path | str | None = None) -> list[dict[str, Any]]:
    directory = _scenario_dir(storage_dir)
    if not directory.exists():
        return []

    metadata: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        record = load_scenario(path.stem, storage_dir=directory)
        metadata.append(
            {
                "scenario_id": record["scenario_id"],
                "scenario_name": record["scenario_name"],
                "created_at": record["created_at"],
                "updated_at": record["updated_at"],
                "override_count": sum(len(section) for section in record["overrides"].values()),
            }
        )
    return sorted(metadata, key=lambda row: (row["scenario_name"].lower(), row["scenario_id"]))


def delete_scenario(scenario_id: str, storage_dir: Path | str | None = None) -> bool:
    path = _scenario_path(scenario_id, storage_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def get_baseline_state() -> dict[str, Any]:
    return {
        "scenario_id": "baseline",
        "scenario_name": "Baseline",
        "overrides": build_override_state(),
        "baseline_source": {
            "config_dir": str(BASE_DIR / "config"),
            "data_dir": str(BASE_DIR / "data"),
        },
    }


def reset_to_baseline() -> dict[str, Any]:
    return get_baseline_state()


def run_saved_scenario(
    scenario_id: str,
    *,
    storage_dir: Path | str | None = None,
    include_deltas: bool = True,
) -> dict[str, Any]:
    record = load_scenario(scenario_id, storage_dir=storage_dir)
    overrides = record["overrides"]
    simulation_result = run_simulation(
        scenario_name=record["scenario_name"],
        class_expected_hours_overrides=overrides["class_expected_hours_overrides"],
        class_weight_overrides=overrides["class_weight_overrides"],
        adjustment_multiplier_overrides=overrides["adjustment_multiplier_overrides"],
    )

    result = {
        "scenario": record,
        "simulation_result": simulation_result,
    }
    if include_deltas:
        result["deltas"] = build_simulation_deltas(
            simulation_result["baseline"], simulation_result["simulated"]
        )
    return result


def compare_saved_scenarios(
    scenario_id_a: str,
    scenario_id_b: str,
    *,
    storage_dir: Path | str | None = None,
) -> dict[str, Any]:
    run_a = run_saved_scenario(scenario_id_a, storage_dir=storage_dir, include_deltas=False)
    run_b = run_saved_scenario(scenario_id_b, storage_dir=storage_dir, include_deltas=False)
    return {
        "scenario_a": run_a["scenario"],
        "scenario_b": run_b["scenario"],
        "simulated_bundle_deltas": build_simulation_deltas(
            run_a["simulation_result"]["simulated"],
            run_b["simulation_result"]["simulated"],
        ),
    }
