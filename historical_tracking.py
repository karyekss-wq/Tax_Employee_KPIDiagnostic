from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "history"

METRIC_FIELDS = [
    "final_score",
    "performance_index",
    "output_score",
    "efficiency_score",
    "accuracy_score",
    "contribution_modifier",
]

SNAPSHOT_KEYS = [
    "run_id",
    "run_name",
    "created_at",
    "source_type",
    "scenario_id",
    "intern_metrics",
]

SOURCE_TYPES = {"baseline", "scenario"}
SAFE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _history_dir(storage_dir: Path | str | None = None) -> Path:
    return Path(storage_dir) if storage_dir is not None else HISTORY_DIR


def make_run_id(run_name: str) -> str:
    """
    Build a deterministic filesystem-safe run id from a display name.
    """
    if not isinstance(run_name, str) or run_name.strip() == "":
        raise ValueError("run_name cannot be blank.")
    run_id = re.sub(r"[^a-z0-9]+", "_", run_name.strip().lower())
    run_id = re.sub(r"_+", "_", run_id).strip("_")
    if run_id == "":
        raise ValueError("run_name collapses to an empty run_id.")
    return run_id


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or run_id.strip() == "":
        raise ValueError("run_id cannot be blank.")
    if run_id != run_id.strip():
        raise ValueError("run_id must not contain leading or trailing whitespace.")
    if not SAFE_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain only lowercase letters, numbers, and underscores.")
    return run_id


def _snapshot_path(run_id: str, storage_dir: Path | str | None = None) -> Path:
    safe_id = validate_run_id(run_id)
    return _history_dir(storage_dir) / f"{safe_id}.json"


def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing required key '{key}'.")
    return mapping[key]


def _require_numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def _validate_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{label} must be a non-empty ISO timestamp string.")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO timestamp string.") from exc
    return value


def _direction(from_value: float, to_value: float) -> str:
    if to_value > from_value:
        return "increase"
    if to_value < from_value:
        return "decrease"
    return "no_change"


def _validate_source(source_type: Any, scenario_id: Any) -> tuple[str, str | None]:
    if source_type not in SOURCE_TYPES:
        raise ValueError(f"source_type must be one of {sorted(SOURCE_TYPES)}.")
    if source_type == "baseline" and scenario_id is not None:
        raise ValueError("baseline source_type requires scenario_id to be None.")
    if source_type == "scenario":
        if not isinstance(scenario_id, str) or scenario_id.strip() == "":
            raise ValueError("scenario source_type requires a non-empty scenario_id.")
        return source_type, scenario_id
    return source_type, None


def _validate_intern_metric(row: Any, label: str) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError(f"{label} must be a dict.")
    required = set(["intern_id", "performance_category"] + METRIC_FIELDS)
    actual = set(row.keys())
    missing = sorted(required - actual)
    unknown = sorted(actual - required)
    if missing:
        raise ValueError(f"{label} missing required field(s): {missing}")
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {unknown}")

    intern_id = row["intern_id"]
    if not isinstance(intern_id, str) or intern_id.strip() == "":
        raise ValueError(f"{label}.intern_id cannot be blank.")
    category = row["performance_category"]
    if not isinstance(category, str) or category.strip() == "":
        raise ValueError(f"{label}.performance_category cannot be blank.")

    validated: dict[str, Any] = {
        "intern_id": intern_id,
        "performance_category": category,
    }
    for metric in METRIC_FIELDS:
        validated[metric] = _require_numeric(row[metric], f"{label}.{metric}")
    return validated


def validate_historical_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise ValueError("historical snapshot must be a dict.")

    actual_keys = set(snapshot.keys())
    expected_keys = set(SNAPSHOT_KEYS)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"historical snapshot missing required key(s): {missing}")
    if unknown:
        raise ValueError(f"historical snapshot has unknown key(s): {unknown}")

    run_name = snapshot["run_name"]
    if not isinstance(run_name, str) or run_name.strip() == "":
        raise ValueError("run_name cannot be blank.")

    source_type, scenario_id = _validate_source(
        snapshot["source_type"], snapshot["scenario_id"]
    )

    intern_metrics = snapshot["intern_metrics"]
    if not isinstance(intern_metrics, list):
        raise ValueError("intern_metrics must be a list.")

    validated_metrics = [
        _validate_intern_metric(row, f"intern_metrics[{idx}]")
        for idx, row in enumerate(intern_metrics)
    ]
    intern_ids = [row["intern_id"] for row in validated_metrics]
    if len(intern_ids) != len(set(intern_ids)):
        raise ValueError("intern_metrics contains duplicate intern_id values.")

    return {
        "run_id": validate_run_id(snapshot["run_id"]),
        "run_name": run_name,
        "created_at": _validate_timestamp(snapshot["created_at"], "created_at"),
        "source_type": source_type,
        "scenario_id": scenario_id,
        "intern_metrics": sorted(validated_metrics, key=lambda row: row["intern_id"]),
    }


def build_historical_snapshot(
    *,
    run_name: str,
    pipeline_result_bundle: dict[str, Any],
    run_id: str | None = None,
    source_type: str = "baseline",
    scenario_id: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """
    Extract compact intern-level metrics from an existing pipeline result bundle.
    """
    if not isinstance(run_name, str) or run_name.strip() == "":
        raise ValueError("run_name cannot be blank.")
    if not isinstance(pipeline_result_bundle, dict):
        raise ValueError("pipeline_result_bundle must be a dict.")
    scores = _require_key(pipeline_result_bundle, "scores", "pipeline_result_bundle")
    if not isinstance(scores, dict):
        raise ValueError("pipeline_result_bundle.scores must be a dict.")

    intern_metrics: list[dict[str, Any]] = []
    for intern_id in sorted(scores.keys()):
        summary = getattr(scores[intern_id], "summary", None)
        if not isinstance(summary, dict):
            raise ValueError(f"scores.{intern_id}.summary must be a dict.")
        row = {
            "intern_id": str(_require_key(summary, "intern_id", f"scores.{intern_id}.summary")),
            "performance_category": _require_key(
                summary, "performance_category", f"scores.{intern_id}.summary"
            ),
        }
        for metric in METRIC_FIELDS:
            row[metric] = _require_key(summary, metric, f"scores.{intern_id}.summary")
        intern_metrics.append(row)

    snapshot = {
        "run_id": validate_run_id(run_id) if run_id else make_run_id(run_name),
        "run_name": run_name,
        "created_at": created_at or _now_iso(),
        "source_type": source_type,
        "scenario_id": scenario_id,
        "intern_metrics": intern_metrics,
    }
    return validate_historical_snapshot(snapshot)


def save_historical_snapshot(
    snapshot: dict[str, Any],
    *,
    overwrite: bool = False,
    storage_dir: Path | str | None = None,
) -> dict[str, Any]:
    validated = validate_historical_snapshot(deepcopy(snapshot))
    path = _snapshot_path(validated["run_id"], storage_dir)
    if path.exists() and not overwrite:
        raise ValueError(f"Historical snapshot '{validated['run_id']}' already exists.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return validated


def load_historical_snapshot(
    run_id: str,
    *,
    storage_dir: Path | str | None = None,
) -> dict[str, Any]:
    path = _snapshot_path(run_id, storage_dir)
    if not path.exists():
        raise FileNotFoundError(f"Historical snapshot '{run_id}' does not exist.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Historical snapshot '{run_id}' contains malformed JSON.") from exc
    return validate_historical_snapshot(raw)


def list_historical_snapshots(
    storage_dir: Path | str | None = None,
) -> list[dict[str, Any]]:
    directory = _history_dir(storage_dir)
    if not directory.exists():
        return []

    metadata: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        snapshot = load_historical_snapshot(path.stem, storage_dir=directory)
        metadata.append(
            {
                "run_id": snapshot["run_id"],
                "run_name": snapshot["run_name"],
                "created_at": snapshot["created_at"],
                "source_type": snapshot["source_type"],
                "scenario_id": snapshot["scenario_id"],
                "intern_count": len(snapshot["intern_metrics"]),
            }
        )
    return sorted(metadata, key=lambda row: (row["run_name"].lower(), row["run_id"]))


def delete_historical_snapshot(
    run_id: str,
    *,
    storage_dir: Path | str | None = None,
) -> bool:
    path = _snapshot_path(run_id, storage_dir)
    if not path.exists():
        return False
    path.unlink()
    return True


def _metrics_by_intern(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["intern_id"]: row for row in snapshot["intern_metrics"]}


def classify_intern_trends(
    from_snapshot: dict[str, Any],
    to_snapshot: dict[str, Any],
) -> list[dict[str, Any]]:
    from_valid = validate_historical_snapshot(deepcopy(from_snapshot))
    to_valid = validate_historical_snapshot(deepcopy(to_snapshot))
    from_by_intern = _metrics_by_intern(from_valid)
    to_by_intern = _metrics_by_intern(to_valid)

    trends: list[dict[str, Any]] = []
    for intern_id in sorted(set(from_by_intern) | set(to_by_intern)):
        if intern_id not in from_by_intern:
            label = "new_in_period"
            from_value = None
            to_value = to_by_intern[intern_id]["final_score"]
        elif intern_id not in to_by_intern:
            label = "missing_in_period"
            from_value = from_by_intern[intern_id]["final_score"]
            to_value = None
        else:
            from_value = from_by_intern[intern_id]["final_score"]
            to_value = to_by_intern[intern_id]["final_score"]
            direction = _direction(float(from_value), float(to_value))
            label = {
                "increase": "improving",
                "decrease": "deteriorating",
                "no_change": "unchanged",
            }[direction]

        trends.append(
            {
                "intern_id": intern_id,
                "trend_label": label,
                "from_final_score": from_value,
                "to_final_score": to_value,
            }
        )
    return trends


def compare_historical_snapshots(
    run_id_a: str,
    run_id_b: str,
    *,
    storage_dir: Path | str | None = None,
) -> dict[str, Any]:
    from_snapshot = load_historical_snapshot(run_id_a, storage_dir=storage_dir)
    to_snapshot = load_historical_snapshot(run_id_b, storage_dir=storage_dir)
    from_by_intern = _metrics_by_intern(from_snapshot)
    to_by_intern = _metrics_by_intern(to_snapshot)

    metric_trends: list[dict[str, Any]] = []
    category_transitions: list[dict[str, Any]] = []
    for intern_id in sorted(set(from_by_intern) & set(to_by_intern)):
        from_row = from_by_intern[intern_id]
        to_row = to_by_intern[intern_id]
        for metric in METRIC_FIELDS:
            from_value = float(from_row[metric])
            to_value = float(to_row[metric])
            metric_trends.append(
                {
                    "intern_id": intern_id,
                    "metric_name": metric,
                    "from_value": from_value,
                    "to_value": to_value,
                    "absolute_delta": to_value - from_value,
                    "direction": _direction(from_value, to_value),
                }
            )
        category_transitions.append(
            {
                "intern_id": intern_id,
                "from_category": from_row["performance_category"],
                "to_category": to_row["performance_category"],
                "changed": from_row["performance_category"] != to_row["performance_category"],
            }
        )

    trend_classifications = classify_intern_trends(from_snapshot, to_snapshot)
    trend_summary = {
        "improving_count": sum(1 for row in trend_classifications if row["trend_label"] == "improving"),
        "deteriorating_count": sum(
            1 for row in trend_classifications if row["trend_label"] == "deteriorating"
        ),
        "unchanged_count": sum(1 for row in trend_classifications if row["trend_label"] == "unchanged"),
        "new_in_period_count": sum(
            1 for row in trend_classifications if row["trend_label"] == "new_in_period"
        ),
        "missing_in_period_count": sum(
            1 for row in trend_classifications if row["trend_label"] == "missing_in_period"
        ),
    }

    return {
        "from_run_id": from_snapshot["run_id"],
        "to_run_id": to_snapshot["run_id"],
        "metric_trends": metric_trends,
        "category_transitions": category_transitions,
        "trend_classifications": trend_classifications,
        "trend_summary": trend_summary,
    }
