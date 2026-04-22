from __future__ import annotations

from typing import Any

from cross_intern_patterns import build_cross_intern_patterns
from diagnostic_insights import build_diagnostic_insights


REQUIRED_ACTION_FIELDS = [
    "action_key",
    "action_type",
    "target_scope",
    "target_id",
    "priority_level",
    "evidence_sources",
    "trigger_type",
    "message",
    "rationale",
    "related_metric_source",
]

PRIORITY_ORDER = {"high": 0, "moderate": 1, "low": 2}


def _priority_from_intern_signal(direction: str, severity: str) -> str:
    if direction in {"drag", "weakness"}:
        if severity == "high":
            return "high"
        if severity == "moderate":
            return "moderate"
        return "low"
    if direction in {"support", "strength"}:
        return "low"
    return "low"


def _priority_from_pattern(pattern: dict[str, Any]) -> str:
    direction = str(pattern.get("direction", "neutral"))
    scope = str(pattern.get("scope_classification", "isolated"))

    if direction in {"drag", "weakness"}:
        if scope == "systemic":
            return "high"
        if scope == "emerging":
            return "moderate"
        return "low"

    if direction in {"support", "strength"}:
        if scope == "systemic":
            return "moderate"
        return "low"

    return "low"


def map_normalized_insight_to_action(
    *,
    intern_id: str,
    normalized_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    strength = normalized_by_key.get("primary_strength_driver")
    weakness = normalized_by_key.get("primary_weakness_driver")
    dominant = normalized_by_key.get("dominant_final_score_driver")
    efficiency = normalized_by_key.get("efficiency_explanation")
    accuracy = normalized_by_key.get("accuracy_explanation")
    contribution = normalized_by_key.get("contribution_explanation")

    if not weakness:
        return actions

    weakness_metric = str(weakness.get("metric_source", ""))
    weakness_severity = str(weakness.get("severity", "low"))

    if weakness.get("direction") == "weakness":
        if weakness_metric == "accuracy_score":
            major_error_trigger = bool(
                dominant
                and str(dominant.get("metric_source")) == "accuracy_attribution"
                and str(dominant.get("evidence_unit")) == "weighted_error_impact"
            )
            trigger = "major_error_concentration" if major_error_trigger else "accuracy_drag"
            priority = "high" if major_error_trigger or weakness_severity == "high" else "moderate"

            actions.append(
                {
                    "action_key": f"intern_review_focus_accuracy_{intern_id}",
                    "action_type": "review_focus",
                    "target_scope": "intern",
                    "target_id": intern_id,
                    "priority_level": priority,
                    "evidence_sources": ["normalized_insights", "accuracy_attribution"],
                    "trigger_type": trigger,
                    "message": (
                        f"Increase review focus for {intern_id} because accuracy is the primary weakness "
                        "and error burden is constraining execution."
                    ),
                    "rationale": (
                        f"Primary weakness metric is accuracy_score (peer gap {float(weakness['evidence_value']):.4f}); "
                        f"dominant driver evidence is {dominant.get('evidence_label') if dominant else 'accuracy drag'}"
                    ),
                    "related_metric_source": "accuracy_score",
                    "intern_id": intern_id,
                    "related_insight_keys": [
                        "primary_weakness_driver",
                        "dominant_final_score_driver",
                        "accuracy_explanation",
                    ],
                    "evidence_value": float(weakness.get("evidence_value", 0.0)),
                    "evidence_unit": str(weakness.get("evidence_unit", "peer_gap")),
                }
            )

        elif weakness_metric == "efficiency_score" and efficiency and efficiency.get("direction") == "drag":
            priority = _priority_from_intern_signal("drag", str(efficiency.get("severity", "low")))
            actions.append(
                {
                    "action_key": f"intern_workload_monitor_efficiency_{intern_id}",
                    "action_type": "workload_monitor",
                    "target_scope": "intern",
                    "target_id": intern_id,
                    "priority_level": priority,
                    "evidence_sources": ["normalized_insights", "efficiency_attribution"],
                    "trigger_type": "efficiency_drag",
                    "message": (
                        f"Monitor workload pacing for {intern_id} because efficiency drag is a primary weakness."
                    ),
                    "rationale": (
                        f"Primary weakness metric is efficiency_score (peer gap {float(weakness['evidence_value']):.4f}) "
                        f"with efficiency evidence: {efficiency.get('evidence_label')}"
                    ),
                    "related_metric_source": "efficiency_score",
                    "intern_id": intern_id,
                    "related_insight_keys": ["primary_weakness_driver", "efficiency_explanation"],
                    "evidence_value": float(efficiency.get("evidence_value", 0.0)),
                    "evidence_unit": str(efficiency.get("evidence_unit", "none")),
                    "task_class": efficiency.get("related_task_class"),
                    "task_id": efficiency.get("related_task_id"),
                }
            )

        elif (
            weakness_metric == "contribution_modifier"
            or (contribution and contribution.get("direction") == "drag")
        ):
            contrib_severity = str((contribution or {}).get("severity", weakness_severity))
            priority = _priority_from_intern_signal("drag", contrib_severity)
            actions.append(
                {
                    "action_key": f"intern_coaching_focus_contribution_{intern_id}",
                    "action_type": "coaching_focus",
                    "target_scope": "intern",
                    "target_id": intern_id,
                    "priority_level": priority,
                    "evidence_sources": ["normalized_insights", "contribution_attribution"],
                    "trigger_type": "contribution_drag",
                    "message": (
                        f"Apply coaching focus for {intern_id} because contribution drag is reducing overall execution."
                    ),
                    "rationale": (
                        f"Primary weakness metric is {weakness_metric}; contribution evidence: "
                        f"{(contribution or {}).get('evidence_label', 'contribution drag')}"
                    ),
                    "related_metric_source": "contribution_modifier",
                    "intern_id": intern_id,
                    "related_insight_keys": ["primary_weakness_driver", "contribution_explanation"],
                    "evidence_value": float((contribution or {}).get("evidence_value", weakness.get("evidence_value", 0.0))),
                    "evidence_unit": str((contribution or {}).get("evidence_unit", weakness.get("evidence_unit", "peer_gap"))),
                    "flag_type": (contribution or {}).get("related_flag_type"),
                }
            )

        else:
            # Isolated or weak signal fallback.
            actions.append(
                {
                    "action_key": f"intern_watchlist_{weakness_metric}_{intern_id}",
                    "action_type": "watchlist",
                    "target_scope": "intern",
                    "target_id": intern_id,
                    "priority_level": "low",
                    "evidence_sources": ["normalized_insights"],
                    "trigger_type": "isolated_weakness",
                    "message": (
                        f"Place {intern_id} on low-intensity watchlist for {weakness_metric} because a weak drag signal is present."
                    ),
                    "rationale": (
                        f"Primary weakness metric is {weakness_metric} with evidence value "
                        f"{float(weakness.get('evidence_value', 0.0)):.4f}."
                    ),
                    "related_metric_source": weakness_metric,
                    "intern_id": intern_id,
                    "related_insight_keys": ["primary_weakness_driver"],
                    "evidence_value": float(weakness.get("evidence_value", 0.0)),
                    "evidence_unit": str(weakness.get("evidence_unit", "peer_gap")),
                }
            )

    # Preserve/recognition signal if there is actionable strength and no high-priority drag action.
    high_drag_exists = any(action["priority_level"] == "high" for action in actions)
    if strength and strength.get("direction") == "strength" and not high_drag_exists:
        strength_metric = str(strength.get("metric_source", ""))
        action_type = "recognition_signal" if str(strength.get("severity")) in {"moderate", "high"} else "preserve_strength"
        actions.append(
            {
                "action_key": f"intern_{action_type}_{strength_metric}_{intern_id}",
                "action_type": action_type,
                "target_scope": "intern",
                "target_id": intern_id,
                "priority_level": "low",
                "evidence_sources": ["normalized_insights"],
                "trigger_type": "output_support" if strength_metric == "output_score" else "positive_contribution_support",
                "message": (
                    f"Preserve current execution approach for {intern_id} because {strength_metric} is a stable support area."
                ),
                "rationale": (
                    f"Primary strength metric is {strength_metric} with peer-gap evidence "
                    f"{float(strength.get('evidence_value', 0.0)):.4f}."
                ),
                "related_metric_source": strength_metric,
                "intern_id": intern_id,
                "related_insight_keys": ["primary_strength_driver"],
                "evidence_value": float(strength.get("evidence_value", 0.0)),
                "evidence_unit": str(strength.get("evidence_unit", "peer_gap")),
            }
        )

    return actions


def map_system_pattern_to_action(pattern: dict[str, Any]) -> dict[str, Any] | None:
    pattern_type = str(pattern.get("pattern_type", ""))
    metric_source = str(pattern.get("metric_source", ""))
    direction = str(pattern.get("direction", "neutral"))

    # Team drag actions.
    if pattern_type == "recurring_weakness" and metric_source == "accuracy_score":
        action_type = "calibration_review"
        trigger_type = "recurring_accuracy_drag"
        target_scope = "team"
        target_id = "accuracy_score"
        message = "Prioritize team calibration review because accuracy weakness recurs across interns."
        rationale = (
            f"Pattern {pattern['pattern_key']} appears at frequency {float(pattern['frequency']):.2f} "
            f"with scope {pattern.get('scope_classification')}"
        )
    elif (
        pattern_type == "recurring_task_class_pattern"
        and metric_source == "efficiency_attribution"
        and direction == "drag"
    ):
        action_type = "training_focus"
        trigger_type = "class_specific_efficiency_drag"
        target_scope = "team"
        target_id = f"class_{pattern.get('task_class')}"
        message = (
            f"Prioritize efficiency training for class {pattern.get('task_class')} work because overrun drag recurs."
        )
        rationale = (
            f"Task-class efficiency drag pattern frequency is {float(pattern['frequency']):.2f} "
            f"({pattern['intern_count']}/{pattern['total_interns']})."
        )
    elif pattern_type == "recurring_flag_pattern" and direction == "drag":
        action_type = "process_review"
        trigger_type = "flag_driven_contribution_drag"
        target_scope = "team"
        target_id = str(pattern.get("flag_type", "unknown_flag"))
        message = (
            f"Review contribution process around {pattern.get('flag_type')} because contribution drag recurs."
        )
        rationale = (
            f"Flag-driven contribution drag pattern frequency is {float(pattern['frequency']):.2f} "
            f"with scope {pattern.get('scope_classification')}."
        )
    elif pattern_type == "recurring_positioning_drag" and metric_source == "performance_index":
        action_type = "team_watchlist"
        trigger_type = "recurring_efficiency_drag"
        target_scope = "system"
        target_id = "performance_index"
        message = "Maintain a team watchlist because below-peer performance_index positioning recurs."
        rationale = (
            f"Positioning drag recurs at frequency {float(pattern['frequency']):.2f} "
            f"({pattern['intern_count']}/{pattern['total_interns']})."
        )
    # Team support actions.
    elif (
        pattern_type in {
        "recurring_strength",
        "recurring_attribution_support",
        "recurring_positioning_support",
        "recurring_adjustment_pattern",
        }
        and direction in {"support", "strength"}
        and metric_source
        in {
            "output_score",
            "output_attribution",
            "efficiency_score",
            "efficiency_attribution",
            "contribution_modifier",
            "contribution_attribution",
        }
    ):
        action_type = "preserve_best_practice"
        trigger_type = "recurring_output_support" if metric_source in {"output_score", "output_attribution"} else "positive_contribution_support"
        target_scope = "team"
        target_id = metric_source
        message = (
            f"Preserve best practice for {metric_source} because support patterns recur across the intern group."
        )
        rationale = (
            f"Support pattern {pattern['pattern_key']} frequency is {float(pattern['frequency']):.2f} "
            f"with scope {pattern.get('scope_classification')}."
        )
    else:
        return None

    priority = _priority_from_pattern(pattern)

    action: dict[str, Any] = {
        "action_key": f"{action_type}_{target_scope}_{target_id}",
        "action_type": action_type,
        "target_scope": target_scope,
        "target_id": str(target_id),
        "priority_level": priority,
        "evidence_sources": ["system_patterns", str(pattern.get("supporting_reference", "system_patterns"))],
        "trigger_type": trigger_type,
        "message": message,
        "rationale": rationale,
        "related_metric_source": metric_source,
        "related_pattern_keys": [str(pattern.get("pattern_key"))],
        "supporting_intern_ids": list(pattern.get("sample_intern_ids", [])),
        "evidence_value": float(pattern.get("frequency", 0.0)),
        "evidence_unit": "frequency",
    }

    if pattern.get("task_class"):
        action["task_class"] = str(pattern.get("task_class"))
    if pattern.get("flag_type"):
        action["flag_type"] = str(pattern.get("flag_type"))
    if pattern.get("adjustment_code"):
        action["adjustment_code"] = str(pattern.get("adjustment_code"))

    return action


def deduplicate_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for action in actions:
        key = str(action.get("action_key", ""))
        if key and key not in deduped:
            deduped[key] = action
    return list(deduped.values())


def _sort_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        actions,
        key=lambda action: (
            PRIORITY_ORDER.get(str(action.get("priority_level", "low")), 99),
            str(action.get("target_scope", "")),
            str(action.get("action_type", "")),
            str(action.get("target_id", "")),
            str(action.get("action_key", "")),
        ),
    )


def build_intern_manager_actions(results_by_intern: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for intern_id in sorted(results_by_intern.keys()):
        insights = build_diagnostic_insights(results_by_intern, intern_id)
        normalized = insights["normalized_insights"]
        normalized_by_key = {record["insight_key"]: record for record in normalized}
        actions.extend(
            map_normalized_insight_to_action(
                intern_id=str(intern_id),
                normalized_by_key=normalized_by_key,
            )
        )

    actions = deduplicate_actions(actions)
    return _sort_actions(actions)


def build_team_manager_actions(results_by_intern: dict[str, Any]) -> list[dict[str, Any]]:
    pattern_payload = build_cross_intern_patterns(results_by_intern)
    actions: list[dict[str, Any]] = []
    for pattern in pattern_payload["system_patterns"]:
        action = map_system_pattern_to_action(pattern)
        if action:
            actions.append(action)

    actions = deduplicate_actions(actions)
    return _sort_actions(actions)


def summarize_manager_actions(intern_actions: list[dict[str, Any]], team_actions: list[dict[str, Any]]) -> dict[str, int]:
    all_actions = intern_actions + team_actions
    return {
        "total_intern_actions": len(intern_actions),
        "total_team_actions": len(team_actions),
        "high_priority_count": sum(1 for action in all_actions if action.get("priority_level") == "high"),
        "moderate_priority_count": sum(
            1 for action in all_actions if action.get("priority_level") == "moderate"
        ),
        "low_priority_count": sum(1 for action in all_actions if action.get("priority_level") == "low"),
    }


def build_manager_actions(results_by_intern: dict[str, Any]) -> dict[str, Any]:
    intern_actions = build_intern_manager_actions(results_by_intern)
    team_actions = build_team_manager_actions(results_by_intern)

    payload = {
        "intern_actions": intern_actions,
        "team_actions": team_actions,
        "action_summary": summarize_manager_actions(intern_actions, team_actions),
    }

    for action in payload["intern_actions"] + payload["team_actions"]:
        for field in REQUIRED_ACTION_FIELDS:
            if field not in action:
                raise ValueError(f"Action missing required field '{field}': {action}")
        if not action.get("evidence_sources"):
            raise ValueError(f"Action has no evidence sources: {action}")

    return payload
