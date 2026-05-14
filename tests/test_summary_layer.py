from __future__ import annotations

import copy

import pytest

from summary_layer import (
    build_decision_summary_narrative,
    summarize_historical_comparison,
    summarize_manager_actions,
    summarize_simulation_deltas,
    summarize_system_patterns,
    validate_summary_bundle,
    validate_summary_object,
)


def _manager_actions() -> dict:
    return {
        "intern_actions": [
            {
                "action_key": "a1",
                "priority_level": "high",
                "target_scope": "intern",
                "action_type": "review_focus",
            },
            {
                "action_key": "a2",
                "priority_level": "low",
                "target_scope": "intern",
                "action_type": "watchlist",
            },
        ],
        "team_actions": [
            {
                "action_key": "a3",
                "priority_level": "high",
                "target_scope": "team",
                "action_type": "training_focus",
            },
            {
                "action_key": "a4",
                "priority_level": "moderate",
                "target_scope": "system",
                "action_type": "calibration_review",
            },
        ],
    }


def _patterns() -> dict:
    return {
        "system_patterns": [
            {
                "pattern_key": "p1",
                "scope_classification": "systemic",
                "frequency": 0.75,
            },
            {
                "pattern_key": "p2",
                "scope_classification": "emerging",
                "frequency": 0.5,
            },
            {
                "pattern_key": "p3",
                "scope_classification": "isolated",
                "frequency": 0.25,
            },
        ]
    }


def _deltas() -> dict:
    return {
        "metric_deltas": [
            {"intern_id": "INT001", "metric_name": "final_score", "direction": "increase"},
            {"intern_id": "INT002", "metric_name": "final_score", "direction": "decrease"},
            {"intern_id": "INT003", "metric_name": "final_score", "direction": "no_change"},
            {"intern_id": "INT001", "metric_name": "accuracy_score", "direction": "increase"},
        ],
        "category_changes": [
            {"intern_id": "INT001", "changed": True},
            {"intern_id": "INT002", "changed": False},
        ],
        "action_changes": [
            {"action_key": "a1", "change_type": "added"},
            {"action_key": "a2", "change_type": "removed"},
            {"action_key": "a3", "change_type": "priority_changed"},
            {"action_key": "a4", "change_type": "unchanged"},
        ],
        "pattern_changes": [
            {"pattern_key": "p1", "change_type": "introduced"},
            {"pattern_key": "p2", "change_type": "resolved"},
            {"pattern_key": "p3", "change_type": "scope_changed"},
            {"pattern_key": "p4", "change_type": "frequency_changed"},
        ],
    }


def _historical_comparison() -> dict:
    return {
        "from_run_id": "week_1",
        "to_run_id": "week_2",
        "trend_summary": {
            "improving_count": 2,
            "deteriorating_count": 1,
            "unchanged_count": 3,
            "new_in_period_count": 1,
            "missing_in_period_count": 0,
        },
        "category_transitions": [
            {"intern_id": "INT001", "changed": True},
            {"intern_id": "INT002", "changed": False},
        ],
    }


def _fact_value(summaries: list[dict], summary_id: str, field: str):
    summary = next(row for row in summaries if row["summary_id"] == summary_id)
    return next(fact["value"] for fact in summary["facts_used"] if fact["field"] == field)


def test_manager_action_summary_counts_high_priority_actions_correctly() -> None:
    summaries = summarize_manager_actions(_manager_actions())
    assert _fact_value(
        summaries, "manager_actions_priority_summary", "high_priority_action_count"
    ) == 2


def test_manager_action_summary_counts_target_scopes_correctly() -> None:
    summaries = summarize_manager_actions(_manager_actions())
    assert _fact_value(summaries, "manager_actions_scope_summary", "scope_counts") == {
        "intern": 2,
        "team": 1,
        "system": 1,
    }


def test_system_pattern_summary_counts_scopes_correctly() -> None:
    summaries = summarize_system_patterns(_patterns())
    assert _fact_value(summaries, "system_patterns_scope_summary", "scope_counts") == {
        "systemic": 1,
        "emerging": 1,
        "isolated": 1,
    }


def test_simulation_delta_summary_counts_final_score_directions_correctly() -> None:
    summaries = summarize_simulation_deltas(_deltas())
    assert _fact_value(
        summaries, "simulation_delta_final_score_summary", "final_score_direction_counts"
    ) == {"increase": 1, "decrease": 1, "no_change": 1}


def test_simulation_delta_summary_counts_category_transitions_correctly() -> None:
    summaries = summarize_simulation_deltas(_deltas())
    assert _fact_value(
        summaries, "simulation_delta_category_summary", "category_transition_count"
    ) == 1


def test_simulation_delta_summary_counts_action_changes_correctly() -> None:
    summaries = summarize_simulation_deltas(_deltas())
    assert _fact_value(summaries, "simulation_delta_action_summary", "action_change_counts") == {
        "added": 1,
        "removed": 1,
        "priority_changed": 1,
    }


def test_simulation_delta_summary_counts_pattern_changes_correctly() -> None:
    summaries = summarize_simulation_deltas(_deltas())
    assert _fact_value(summaries, "simulation_delta_pattern_summary", "pattern_change_counts") == {
        "introduced": 1,
        "resolved": 1,
        "scope_changed": 1,
        "frequency_changed": 1,
    }


def test_historical_comparison_summary_reports_trend_counts_correctly() -> None:
    summaries = summarize_historical_comparison(_historical_comparison())
    assert _fact_value(summaries, "historical_trend_summary", "trend_summary_counts") == {
        "improving_count": 2,
        "deteriorating_count": 1,
        "unchanged_count": 3,
        "new_in_period_count": 1,
        "missing_in_period_count": 0,
    }


def test_summary_object_validation_catches_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="missing required key"):
        validate_summary_object({"summary_id": "x"})


def test_summary_object_validation_catches_blank_summary_text() -> None:
    summary = summarize_manager_actions(_manager_actions())[0]
    summary["summary_text"] = " "
    with pytest.raises(ValueError, match="summary_text"):
        validate_summary_object(summary)


def test_summary_bundle_validation_validates_all_objects() -> None:
    summaries = summarize_manager_actions(_manager_actions())
    assert validate_summary_bundle(summaries) == summaries


def test_summaries_do_not_mutate_input_structures() -> None:
    actions = _manager_actions()
    patterns = _patterns()
    deltas = _deltas()
    historical = _historical_comparison()
    before = copy.deepcopy((actions, patterns, deltas, historical))

    summarize_manager_actions(actions)
    summarize_system_patterns(patterns)
    summarize_simulation_deltas(deltas)
    summarize_historical_comparison(historical)

    assert (actions, patterns, deltas, historical) == before


def test_empty_inputs_produce_explicit_no_data_summaries() -> None:
    action_summaries = summarize_manager_actions({"intern_actions": [], "team_actions": []})
    pattern_summaries = summarize_system_patterns({"system_patterns": []})
    assert "0 manager actions are currently high priority." in action_summaries[0]["summary_text"]
    assert "No system patterns are present." == pattern_summaries[1]["summary_text"]


def test_generated_by_is_deterministic_template_by_default() -> None:
    summaries = (
        summarize_manager_actions(_manager_actions())
        + summarize_system_patterns(_patterns())
        + summarize_simulation_deltas(_deltas())
        + summarize_historical_comparison(_historical_comparison())
    )
    assert {summary["generated_by"] for summary in summaries} == {"deterministic_template"}


def test_facts_used_supports_summary_counts() -> None:
    summaries = summarize_simulation_deltas(_deltas())
    summary = next(
        row for row in summaries if row["summary_id"] == "simulation_delta_category_summary"
    )
    assert summary["facts_used"] == [{"field": "category_transition_count", "value": 1}]
    assert "1 performance category transition" in summary["summary_text"]


def test_no_external_api_or_network_dependency_is_required() -> None:
    summaries = summarize_manager_actions(_manager_actions())
    assert summaries


def test_manager_decision_view_integration_can_import_summary_functions_cleanly() -> None:
    from manager_decision_view import render_summary_objects

    assert callable(render_summary_objects)


def test_decision_summary_narrative_uses_only_summary_objects() -> None:
    summaries = summarize_manager_actions(_manager_actions())[:2]
    narrative = build_decision_summary_narrative(summaries)
    assert narrative["source_keys"] == [summary["summary_id"] for summary in summaries]
    assert narrative["facts_used"] == [
        fact for summary in summaries for fact in summary["facts_used"]
    ]
