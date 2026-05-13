from __future__ import annotations

from numbers import Real
from typing import Any


METRICS_TO_COMPARE = [
    "final_score",
    "performance_index",
    "output_score",
    "efficiency_score",
    "accuracy_score",
    "contribution_modifier",
]

DRIVER_KEYS_TO_COMPARE = [
    "primary_strength_driver",
    "primary_weakness_driver",
    "dominant_final_score_driver",
]


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict.")
    return value


def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{label} is missing required key '{key}'.")
    return mapping[key]


def _require_numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def _direction_from_delta(delta: float) -> str:
    if delta > 0:
        return "increase"
    if delta < 0:
        return "decrease"
    return "no_change"


def _get_scores(bundle: dict[str, Any], label: str) -> dict[str, Any]:
    return _require_mapping(_require_key(bundle, "scores", label), f"{label}.scores")


def _get_diagnostics(bundle: dict[str, Any], label: str) -> dict[str, Any]:
    return _require_mapping(
        _require_key(bundle, "diagnostics", label), f"{label}.diagnostics"
    )


def _validate_same_keys(
    baseline_items: dict[str, Any],
    simulated_items: dict[str, Any],
    label: str,
) -> list[str]:
    baseline_keys = set(baseline_items.keys())
    simulated_keys = set(simulated_items.keys())
    if baseline_keys != simulated_keys:
        raise ValueError(
            f"{label} keys differ. "
            f"baseline_only={sorted(baseline_keys - simulated_keys)}, "
            f"simulated_only={sorted(simulated_keys - baseline_keys)}"
        )
    return sorted(str(key) for key in baseline_keys)


def build_metric_deltas(
    baseline_bundle: dict[str, Any],
    simulated_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_scores = _get_scores(baseline_bundle, "baseline")
    simulated_scores = _get_scores(simulated_bundle, "simulated")
    intern_ids = _validate_same_keys(baseline_scores, simulated_scores, "score")

    deltas: list[dict[str, Any]] = []
    for intern_id in intern_ids:
        baseline_summary = _require_mapping(
            getattr(baseline_scores[intern_id], "summary", None),
            f"baseline.scores.{intern_id}.summary",
        )
        simulated_summary = _require_mapping(
            getattr(simulated_scores[intern_id], "summary", None),
            f"simulated.scores.{intern_id}.summary",
        )
        for metric_name in METRICS_TO_COMPARE:
            baseline_value = _require_numeric(
                _require_key(
                    baseline_summary,
                    metric_name,
                    f"baseline.scores.{intern_id}.summary",
                ),
                f"baseline {intern_id} {metric_name}",
            )
            simulated_value = _require_numeric(
                _require_key(
                    simulated_summary,
                    metric_name,
                    f"simulated.scores.{intern_id}.summary",
                ),
                f"simulated {intern_id} {metric_name}",
            )
            absolute_delta = simulated_value - baseline_value
            deltas.append(
                {
                    "intern_id": intern_id,
                    "metric_name": metric_name,
                    "baseline_value": baseline_value,
                    "simulated_value": simulated_value,
                    "absolute_delta": absolute_delta,
                    "direction": _direction_from_delta(absolute_delta),
                }
            )
    return deltas


def build_category_changes(
    baseline_bundle: dict[str, Any],
    simulated_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_scores = _get_scores(baseline_bundle, "baseline")
    simulated_scores = _get_scores(simulated_bundle, "simulated")
    intern_ids = _validate_same_keys(baseline_scores, simulated_scores, "score")

    changes: list[dict[str, Any]] = []
    for intern_id in intern_ids:
        baseline_summary = _require_mapping(
            getattr(baseline_scores[intern_id], "summary", None),
            f"baseline.scores.{intern_id}.summary",
        )
        simulated_summary = _require_mapping(
            getattr(simulated_scores[intern_id], "summary", None),
            f"simulated.scores.{intern_id}.summary",
        )
        baseline_category = _require_key(
            baseline_summary, "performance_category", f"baseline.scores.{intern_id}.summary"
        )
        simulated_category = _require_key(
            simulated_summary,
            "performance_category",
            f"simulated.scores.{intern_id}.summary",
        )
        changes.append(
            {
                "intern_id": intern_id,
                "baseline_category": str(baseline_category),
                "simulated_category": str(simulated_category),
                "changed": baseline_category != simulated_category,
            }
        )
    return changes


def _normalized_by_key(diagnostics: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    normalized = _require_key(diagnostics, "normalized_insights", label)
    if not isinstance(normalized, list):
        raise ValueError(f"{label}.normalized_insights must be a list.")

    by_key: dict[str, dict[str, Any]] = {}
    for record in normalized:
        row = _require_mapping(record, f"{label}.normalized_insights record")
        insight_key = str(_require_key(row, "insight_key", label))
        if insight_key in by_key:
            raise ValueError(f"{label} has duplicate normalized insight_key '{insight_key}'.")
        by_key[insight_key] = row
    return by_key


def build_driver_changes(
    baseline_bundle: dict[str, Any],
    simulated_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_diagnostics = _get_diagnostics(baseline_bundle, "baseline")
    simulated_diagnostics = _get_diagnostics(simulated_bundle, "simulated")
    intern_ids = _validate_same_keys(baseline_diagnostics, simulated_diagnostics, "diagnostic")

    changes: list[dict[str, Any]] = []
    for intern_id in intern_ids:
        baseline_by_key = _normalized_by_key(
            _require_mapping(
                baseline_diagnostics[intern_id], f"baseline.diagnostics.{intern_id}"
            ),
            f"baseline.diagnostics.{intern_id}",
        )
        simulated_by_key = _normalized_by_key(
            _require_mapping(
                simulated_diagnostics[intern_id], f"simulated.diagnostics.{intern_id}"
            ),
            f"simulated.diagnostics.{intern_id}",
        )
        for driver_type in DRIVER_KEYS_TO_COMPARE:
            baseline_record = _require_mapping(
                _require_key(
                    baseline_by_key,
                    driver_type,
                    f"baseline.diagnostics.{intern_id}.normalized_insights",
                ),
                f"baseline driver {intern_id} {driver_type}",
            )
            simulated_record = _require_mapping(
                _require_key(
                    simulated_by_key,
                    driver_type,
                    f"simulated.diagnostics.{intern_id}.normalized_insights",
                ),
                f"simulated driver {intern_id} {driver_type}",
            )
            baseline_driver = str(
                _require_key(
                    baseline_record,
                    "metric_source",
                    f"baseline driver {intern_id} {driver_type}",
                )
            )
            simulated_driver = str(
                _require_key(
                    simulated_record,
                    "metric_source",
                    f"simulated driver {intern_id} {driver_type}",
                )
            )
            changes.append(
                {
                    "intern_id": intern_id,
                    "driver_type": driver_type,
                    "baseline_driver": baseline_driver,
                    "simulated_driver": simulated_driver,
                    "changed": baseline_driver != simulated_driver,
                }
            )
    return changes


def _manager_actions_by_key(bundle: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    manager_actions = _require_mapping(
        _require_key(bundle, "manager_actions", label), f"{label}.manager_actions"
    )
    actions: list[Any] = []
    for section in ["intern_actions", "team_actions"]:
        section_actions = _require_key(manager_actions, section, f"{label}.manager_actions")
        if not isinstance(section_actions, list):
            raise ValueError(f"{label}.manager_actions.{section} must be a list.")
        actions.extend(section_actions)

    by_key: dict[str, dict[str, Any]] = {}
    for action in actions:
        row = _require_mapping(action, f"{label}.manager_actions action")
        action_key = str(_require_key(row, "action_key", f"{label}.manager_actions action"))
        if action_key in by_key:
            raise ValueError(f"{label}.manager_actions has duplicate action_key '{action_key}'.")
        by_key[action_key] = row
    return by_key


def _action_delta(action_key: str, action: dict[str, Any], change_type: str) -> dict[str, Any]:
    target_scope = str(_require_key(action, "target_scope", f"action {action_key}"))
    intern_id = None
    if target_scope == "intern":
        intern_id = str(_require_key(action, "intern_id", f"action {action_key}"))

    return {
        "action_key": action_key,
        "intern_id": intern_id,
        "action_type": str(_require_key(action, "action_type", f"action {action_key}")),
        "target_scope": target_scope,
        "target_id": str(_require_key(action, "target_id", f"action {action_key}")),
        "baseline_priority": None,
        "simulated_priority": None,
        "change_type": change_type,
    }


def build_action_changes(
    baseline_bundle: dict[str, Any],
    simulated_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_actions = _manager_actions_by_key(baseline_bundle, "baseline")
    simulated_actions = _manager_actions_by_key(simulated_bundle, "simulated")

    changes: list[dict[str, Any]] = []
    for action_key in sorted(set(baseline_actions) | set(simulated_actions)):
        if action_key not in baseline_actions:
            row = _action_delta(action_key, simulated_actions[action_key], "added")
            row["simulated_priority"] = str(
                _require_key(simulated_actions[action_key], "priority_level", f"action {action_key}")
            )
            changes.append(row)
            continue

        if action_key not in simulated_actions:
            row = _action_delta(action_key, baseline_actions[action_key], "removed")
            row["baseline_priority"] = str(
                _require_key(baseline_actions[action_key], "priority_level", f"action {action_key}")
            )
            changes.append(row)
            continue

        baseline_priority = str(
            _require_key(baseline_actions[action_key], "priority_level", f"baseline action {action_key}")
        )
        simulated_priority = str(
            _require_key(simulated_actions[action_key], "priority_level", f"simulated action {action_key}")
        )
        row = _action_delta(
            action_key,
            simulated_actions[action_key],
            "priority_changed" if baseline_priority != simulated_priority else "unchanged",
        )
        row["baseline_priority"] = baseline_priority
        row["simulated_priority"] = simulated_priority
        changes.append(row)
    return changes


def _patterns_by_key(bundle: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    system_patterns = _require_mapping(
        _require_key(bundle, "system_patterns", label), f"{label}.system_patterns"
    )
    patterns = _require_key(system_patterns, "system_patterns", f"{label}.system_patterns")
    if not isinstance(patterns, list):
        raise ValueError(f"{label}.system_patterns.system_patterns must be a list.")

    by_key: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        row = _require_mapping(pattern, f"{label}.system_patterns pattern")
        pattern_key = str(_require_key(row, "pattern_key", f"{label}.system_patterns pattern"))
        if pattern_key in by_key:
            raise ValueError(f"{label}.system_patterns has duplicate pattern_key '{pattern_key}'.")
        by_key[pattern_key] = row
    return by_key


def _pattern_delta(pattern_key: str, pattern: dict[str, Any], change_type: str) -> dict[str, Any]:
    return {
        "pattern_key": pattern_key,
        "pattern_type": str(_require_key(pattern, "pattern_type", f"pattern {pattern_key}")),
        "baseline_scope": None,
        "simulated_scope": None,
        "baseline_frequency": None,
        "simulated_frequency": None,
        "change_type": change_type,
    }


def build_pattern_changes(
    baseline_bundle: dict[str, Any],
    simulated_bundle: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_patterns = _patterns_by_key(baseline_bundle, "baseline")
    simulated_patterns = _patterns_by_key(simulated_bundle, "simulated")

    changes: list[dict[str, Any]] = []
    for pattern_key in sorted(set(baseline_patterns) | set(simulated_patterns)):
        if pattern_key not in baseline_patterns:
            row = _pattern_delta(pattern_key, simulated_patterns[pattern_key], "introduced")
            row["simulated_scope"] = str(
                _require_key(
                    simulated_patterns[pattern_key],
                    "scope_classification",
                    f"pattern {pattern_key}",
                )
            )
            row["simulated_frequency"] = _require_numeric(
                _require_key(simulated_patterns[pattern_key], "frequency", f"pattern {pattern_key}"),
                f"simulated pattern {pattern_key} frequency",
            )
            changes.append(row)
            continue

        if pattern_key not in simulated_patterns:
            row = _pattern_delta(pattern_key, baseline_patterns[pattern_key], "resolved")
            row["baseline_scope"] = str(
                _require_key(
                    baseline_patterns[pattern_key],
                    "scope_classification",
                    f"pattern {pattern_key}",
                )
            )
            row["baseline_frequency"] = _require_numeric(
                _require_key(baseline_patterns[pattern_key], "frequency", f"pattern {pattern_key}"),
                f"baseline pattern {pattern_key} frequency",
            )
            changes.append(row)
            continue

        baseline_scope = str(
            _require_key(
                baseline_patterns[pattern_key],
                "scope_classification",
                f"baseline pattern {pattern_key}",
            )
        )
        simulated_scope = str(
            _require_key(
                simulated_patterns[pattern_key],
                "scope_classification",
                f"simulated pattern {pattern_key}",
            )
        )
        baseline_frequency = _require_numeric(
            _require_key(
                baseline_patterns[pattern_key],
                "frequency",
                f"baseline pattern {pattern_key}",
            ),
            f"baseline pattern {pattern_key} frequency",
        )
        simulated_frequency = _require_numeric(
            _require_key(
                simulated_patterns[pattern_key],
                "frequency",
                f"simulated pattern {pattern_key}",
            ),
            f"simulated pattern {pattern_key} frequency",
        )

        if baseline_scope != simulated_scope:
            change_type = "scope_changed"
        elif baseline_frequency != simulated_frequency:
            change_type = "frequency_changed"
        else:
            change_type = "unchanged"

        row = _pattern_delta(pattern_key, simulated_patterns[pattern_key], change_type)
        row["baseline_scope"] = baseline_scope
        row["simulated_scope"] = simulated_scope
        row["baseline_frequency"] = baseline_frequency
        row["simulated_frequency"] = simulated_frequency
        changes.append(row)
    return changes


def build_simulation_deltas(
    baseline_bundle: dict[str, Any],
    simulated_bundle: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Compare already-generated baseline and simulated pipeline bundles.
    """
    baseline = _require_mapping(baseline_bundle, "baseline_bundle")
    simulated = _require_mapping(simulated_bundle, "simulated_bundle")
    return {
        "metric_deltas": build_metric_deltas(baseline, simulated),
        "category_changes": build_category_changes(baseline, simulated),
        "driver_changes": build_driver_changes(baseline, simulated),
        "action_changes": build_action_changes(baseline, simulated),
        "pattern_changes": build_pattern_changes(baseline, simulated),
    }
