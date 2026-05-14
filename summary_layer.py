from __future__ import annotations

from copy import deepcopy
from typing import Any


SUMMARY_REQUIRED_KEYS = [
    "summary_id",
    "summary_type",
    "source_section",
    "source_keys",
    "summary_text",
    "generated_by",
    "facts_used",
]

GENERATED_BY = "deterministic_template"


def _require_key(mapping: dict[str, Any], key: str, label: str) -> Any:
    if not isinstance(mapping, dict):
        raise ValueError(f"{label} must be a dict.")
    if key not in mapping:
        raise ValueError(f"{label} is missing required key '{key}'.")
    return mapping[key]


def _summary_object(
    *,
    summary_id: str,
    summary_type: str,
    source_section: str,
    source_keys: list[str],
    summary_text: str,
    facts_used: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "summary_id": summary_id,
        "summary_type": summary_type,
        "source_section": source_section,
        "source_keys": source_keys,
        "summary_text": summary_text,
        "generated_by": GENERATED_BY,
        "facts_used": facts_used,
    }
    validate_summary_object(summary)
    return summary


def _fact(field: str, value: Any) -> dict[str, Any]:
    return {"field": field, "value": value}


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def validate_summary_object(summary: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(summary, dict):
        raise ValueError("summary must be a dict.")

    missing = [key for key in SUMMARY_REQUIRED_KEYS if key not in summary]
    if missing:
        raise ValueError(f"summary is missing required key(s): {missing}")

    for key in ["summary_id", "summary_type", "source_section", "generated_by"]:
        if not isinstance(summary[key], str) or summary[key].strip() == "":
            raise ValueError(f"{key} must be a non-empty string.")

    if not isinstance(summary["summary_text"], str) or summary["summary_text"].strip() == "":
        raise ValueError("summary_text must be a non-empty string.")

    if not isinstance(summary["source_keys"], list):
        raise ValueError("source_keys must be a list.")
    for source_key in summary["source_keys"]:
        if not isinstance(source_key, str) or source_key.strip() == "":
            raise ValueError("source_keys must contain only non-empty strings.")

    if not isinstance(summary["facts_used"], list):
        raise ValueError("facts_used must be a list.")
    for fact in summary["facts_used"]:
        if not isinstance(fact, dict):
            raise ValueError("facts_used entries must be dicts.")
        if "field" not in fact or "value" not in fact:
            raise ValueError("facts_used entries must contain field and value.")
        if not isinstance(fact["field"], str) or fact["field"].strip() == "":
            raise ValueError("facts_used field must be a non-empty string.")

    return summary


def validate_summary_bundle(summary_bundle: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(summary_bundle, list):
        raise ValueError("summary_bundle must be a list.")
    return [validate_summary_object(summary) for summary in summary_bundle]


def _manager_action_rows(manager_actions: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(manager_actions, list):
        rows = manager_actions
    elif isinstance(manager_actions, dict):
        intern_actions = _require_key(manager_actions, "intern_actions", "manager_actions")
        team_actions = _require_key(manager_actions, "team_actions", "manager_actions")
        if not isinstance(intern_actions, list) or not isinstance(team_actions, list):
            raise ValueError("manager_actions intern_actions and team_actions must be lists.")
        rows = intern_actions + team_actions
    else:
        raise ValueError("manager_actions must be a dict or list.")

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("manager action rows must be dicts.")
    return [dict(row) for row in rows]


def summarize_manager_actions(manager_actions: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _manager_action_rows(deepcopy(manager_actions))
    source_keys = [str(_require_key(row, "action_key", "manager action")) for row in rows]
    high_count = sum(1 for row in rows if row.get("priority_level") == "high")
    scope_counts = {
        "intern": sum(1 for row in rows if row.get("target_scope") == "intern"),
        "team": sum(1 for row in rows if row.get("target_scope") == "team"),
        "system": sum(1 for row in rows if row.get("target_scope") == "system"),
    }
    action_types = sorted({str(row.get("action_type")) for row in rows if row.get("action_type")})

    return validate_summary_bundle(
        [
            _summary_object(
                summary_id="manager_actions_priority_summary",
                summary_type="manager_actions",
                source_section="manager_actions",
                source_keys=source_keys,
                summary_text=(
                    f"{high_count} manager {_plural(high_count, 'action')} "
                    f"{'is' if high_count == 1 else 'are'} currently high priority."
                ),
                facts_used=[_fact("high_priority_action_count", high_count)],
            ),
            _summary_object(
                summary_id="manager_actions_scope_summary",
                summary_type="manager_actions",
                source_section="manager_actions",
                source_keys=source_keys,
                summary_text=(
                    f"{scope_counts['intern']} intern-scope, {scope_counts['team']} team-scope, "
                    f"and {scope_counts['system']} system-scope manager actions are present."
                ),
                facts_used=[_fact("scope_counts", scope_counts)],
            ),
            _summary_object(
                summary_id="manager_actions_type_summary",
                summary_type="manager_actions",
                source_section="manager_actions",
                source_keys=source_keys,
                summary_text=(
                    "Manager action types represented: "
                    + (", ".join(action_types) if action_types else "none")
                    + "."
                ),
                facts_used=[_fact("action_types", action_types)],
            ),
        ]
    )


def _system_pattern_rows(system_patterns: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(system_patterns, list):
        rows = system_patterns
    elif isinstance(system_patterns, dict):
        rows = _require_key(system_patterns, "system_patterns", "system_patterns")
    else:
        raise ValueError("system_patterns must be a dict or list.")
    if not isinstance(rows, list):
        raise ValueError("system_patterns rows must be a list.")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("system pattern rows must be dicts.")
    return [dict(row) for row in rows]


def summarize_system_patterns(system_patterns: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _system_pattern_rows(deepcopy(system_patterns))
    source_keys = [str(_require_key(row, "pattern_key", "system pattern")) for row in rows]
    scope_counts = {
        "systemic": sum(1 for row in rows if row.get("scope_classification") == "systemic"),
        "emerging": sum(1 for row in rows if row.get("scope_classification") == "emerging"),
        "isolated": sum(1 for row in rows if row.get("scope_classification") == "isolated"),
    }

    most_frequent = None
    if rows:
        most_frequent = max(
            rows,
            key=lambda row: (
                float(_require_key(row, "frequency", "system pattern")),
                str(_require_key(row, "pattern_key", "system pattern")),
            ),
        )
        most_frequent_fact = {
            "pattern_key": most_frequent["pattern_key"],
            "frequency": float(most_frequent["frequency"]),
        }
        most_frequent_text = (
            f"Most frequent pattern is {most_frequent['pattern_key']} "
            f"at frequency {float(most_frequent['frequency']):.4f}."
        )
    else:
        most_frequent_fact = None
        most_frequent_text = "No system patterns are present."

    return validate_summary_bundle(
        [
            _summary_object(
                summary_id="system_patterns_scope_summary",
                summary_type="system_patterns",
                source_section="system_patterns",
                source_keys=source_keys,
                summary_text=(
                    f"{scope_counts['systemic']} systemic, {scope_counts['emerging']} emerging, "
                    f"and {scope_counts['isolated']} isolated patterns are present."
                ),
                facts_used=[_fact("scope_counts", scope_counts)],
            ),
            _summary_object(
                summary_id="system_patterns_frequency_summary",
                summary_type="system_patterns",
                source_section="system_patterns",
                source_keys=source_keys,
                summary_text=most_frequent_text,
                facts_used=[_fact("most_frequent_pattern", most_frequent_fact)],
            ),
        ]
    )


def _count_change_types(rows: list[dict[str, Any]], allowed: list[str]) -> dict[str, int]:
    return {
        change_type: sum(1 for row in rows if row.get("change_type") == change_type)
        for change_type in allowed
    }


def summarize_simulation_deltas(delta_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(delta_bundle, dict):
        raise ValueError("delta_bundle must be a dict.")
    metric_deltas = _require_key(delta_bundle, "metric_deltas", "delta_bundle")
    category_changes = _require_key(delta_bundle, "category_changes", "delta_bundle")
    action_changes = _require_key(delta_bundle, "action_changes", "delta_bundle")
    pattern_changes = _require_key(delta_bundle, "pattern_changes", "delta_bundle")

    final_score_rows = [row for row in metric_deltas if row.get("metric_name") == "final_score"]
    direction_counts = {
        "increase": sum(1 for row in final_score_rows if row.get("direction") == "increase"),
        "decrease": sum(1 for row in final_score_rows if row.get("direction") == "decrease"),
        "no_change": sum(1 for row in final_score_rows if row.get("direction") == "no_change"),
    }
    category_change_count = sum(1 for row in category_changes if row.get("changed") is True)
    action_change_counts = _count_change_types(
        action_changes, ["added", "removed", "priority_changed"]
    )
    pattern_change_counts = _count_change_types(
        pattern_changes, ["introduced", "resolved", "scope_changed", "frequency_changed"]
    )
    source_keys = [
        f"{row.get('intern_id')}:{row.get('metric_name')}"
        for row in final_score_rows
    ]

    return validate_summary_bundle(
        [
            _summary_object(
                summary_id="simulation_delta_final_score_summary",
                summary_type="simulation_delta",
                source_section="metric_deltas",
                source_keys=source_keys,
                summary_text=(
                    f"{direction_counts['increase']} final_score metrics increased, "
                    f"{direction_counts['decrease']} decreased, and "
                    f"{direction_counts['no_change']} had no change in the simulation."
                ),
                facts_used=[_fact("final_score_direction_counts", direction_counts)],
            ),
            _summary_object(
                summary_id="simulation_delta_category_summary",
                summary_type="simulation_delta",
                source_section="category_changes",
                source_keys=[str(row.get("intern_id")) for row in category_changes],
                summary_text=(
                    f"{category_change_count} performance category "
                    f"{_plural(category_change_count, 'transition')} occurred in the simulation."
                ),
                facts_used=[_fact("category_transition_count", category_change_count)],
            ),
            _summary_object(
                summary_id="simulation_delta_action_summary",
                summary_type="simulation_delta",
                source_section="action_changes",
                source_keys=[str(row.get("action_key")) for row in action_changes],
                summary_text=(
                    f"{action_change_counts['added']} actions were added, "
                    f"{action_change_counts['removed']} removed, and "
                    f"{action_change_counts['priority_changed']} changed priority."
                ),
                facts_used=[_fact("action_change_counts", action_change_counts)],
            ),
            _summary_object(
                summary_id="simulation_delta_pattern_summary",
                summary_type="simulation_delta",
                source_section="pattern_changes",
                source_keys=[str(row.get("pattern_key")) for row in pattern_changes],
                summary_text=(
                    f"{pattern_change_counts['introduced']} patterns were introduced, "
                    f"{pattern_change_counts['resolved']} resolved, "
                    f"{pattern_change_counts['scope_changed']} changed scope, and "
                    f"{pattern_change_counts['frequency_changed']} changed frequency."
                ),
                facts_used=[_fact("pattern_change_counts", pattern_change_counts)],
            ),
        ]
    )


def summarize_historical_comparison(comparison_bundle: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(comparison_bundle, dict):
        raise ValueError("comparison_bundle must be a dict.")
    trend_summary = _require_key(comparison_bundle, "trend_summary", "comparison_bundle")
    category_transitions = _require_key(
        comparison_bundle, "category_transitions", "comparison_bundle"
    )
    category_change_count = sum(1 for row in category_transitions if row.get("changed") is True)
    required = [
        "improving_count",
        "deteriorating_count",
        "unchanged_count",
        "new_in_period_count",
        "missing_in_period_count",
    ]
    counts = {key: int(_require_key(trend_summary, key, "trend_summary")) for key in required}

    return validate_summary_bundle(
        [
            _summary_object(
                summary_id="historical_trend_summary",
                summary_type="historical_comparison",
                source_section="trend_summary",
                source_keys=[
                    str(comparison_bundle.get("from_run_id", "")),
                    str(comparison_bundle.get("to_run_id", "")),
                ],
                summary_text=(
                    f"{counts['improving_count']} interns improved, "
                    f"{counts['deteriorating_count']} deteriorated, "
                    f"{counts['unchanged_count']} were unchanged, "
                    f"{counts['new_in_period_count']} were new, and "
                    f"{counts['missing_in_period_count']} were missing in the later period."
                ),
                facts_used=[_fact("trend_summary_counts", counts)],
            ),
            _summary_object(
                summary_id="historical_category_transition_summary",
                summary_type="historical_comparison",
                source_section="category_transitions",
                source_keys=[str(row.get("intern_id")) for row in category_transitions],
                summary_text=(
                    f"{category_change_count} performance category "
                    f"{_plural(category_change_count, 'transition')} occurred between snapshots."
                ),
                facts_used=[_fact("category_transition_count", category_change_count)],
            ),
        ]
    )


def build_decision_summary_narrative(summary_bundle: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = validate_summary_bundle(summary_bundle)
    narrative_text = " ".join(summary["summary_text"] for summary in summaries)
    facts_used = [
        fact
        for summary in summaries
        for fact in summary["facts_used"]
    ]
    return _summary_object(
        summary_id="decision_summary_narrative",
        summary_type="decision_narrative",
        source_section="summary_bundle",
        source_keys=[summary["summary_id"] for summary in summaries],
        summary_text=narrative_text if narrative_text else "No summaries are available.",
        facts_used=facts_used,
    )
