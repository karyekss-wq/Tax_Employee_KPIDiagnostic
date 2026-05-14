from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from manager_decision_view import (
    build_baseline_vs_simulated_rows,
    build_decision_summary,
    build_delta_preview,
    extract_historical_trend_summary,
    group_changes_by_type,
)


def _score(intern_id: str, final_score: float, category: str = "Solid"):
    return SimpleNamespace(
        summary={
            "intern_id": intern_id,
            "final_score": final_score,
            "performance_category": category,
        }
    )


def _simulation_result() -> dict:
    return {
        "baseline": {
            "scores": {
                "INT001": _score("INT001", 80.0, "Risk"),
                "INT002": _score("INT002", 90.0, "Top"),
            }
        },
        "simulated": {
            "scores": {
                "INT001": _score("INT001", 85.0, "Solid"),
                "INT002": _score("INT002", 90.0, "Top"),
            }
        },
    }


def _deltas() -> dict:
    return {
        "metric_deltas": [
            {
                "intern_id": "INT001",
                "metric_name": "final_score",
                "direction": "increase",
                "absolute_delta": 5.0,
            },
            {
                "intern_id": "INT002",
                "metric_name": "final_score",
                "direction": "no_change",
                "absolute_delta": 0.0,
            },
        ],
        "category_changes": [
            {"intern_id": "INT001", "changed": True},
            {"intern_id": "INT002", "changed": False},
        ],
        "driver_changes": [],
        "action_changes": [
            {"action_key": "a1", "change_type": "added"},
            {"action_key": "a2", "change_type": "removed"},
            {"action_key": "a3", "change_type": "unchanged"},
        ],
        "pattern_changes": [
            {"pattern_key": "p1", "change_type": "introduced"},
            {"pattern_key": "p2", "change_type": "scope_changed"},
            {"pattern_key": "p3", "change_type": "unchanged"},
        ],
    }


def test_decision_summary_builder_returns_expected_counts() -> None:
    summary = build_decision_summary(
        results_by_intern={"INT001": _score("INT001", 80.0)},
        patterns_payload={"pattern_summary": {"systemic_count": 2}},
        actions_payload={"action_summary": {"high_priority_count": 3}},
        saved_scenarios=[{"scenario_id": "s1"}, {"scenario_id": "s2"}],
        historical_snapshots=[{"run_id": "r1"}],
    )

    assert summary == {
        "intern_count": 1,
        "highest_priority_action_count": 3,
        "systemic_pattern_count": 2,
        "saved_scenario_count": 2,
        "historical_snapshot_count": 1,
    }


def test_action_change_grouping_groups_by_change_type() -> None:
    grouped = group_changes_by_type(_deltas()["action_changes"])
    assert [row["action_key"] for row in grouped["added"]] == ["a1"]
    assert [row["action_key"] for row in grouped["removed"]] == ["a2"]
    assert [row["action_key"] for row in grouped["unchanged"]] == ["a3"]


def test_pattern_change_grouping_groups_by_change_type() -> None:
    grouped = group_changes_by_type(_deltas()["pattern_changes"])
    assert [row["pattern_key"] for row in grouped["introduced"]] == ["p1"]
    assert [row["pattern_key"] for row in grouped["scope_changed"]] == ["p2"]
    assert [row["pattern_key"] for row in grouped["unchanged"]] == ["p3"]


def test_baseline_vs_simulated_table_builder_preserves_intern_ids() -> None:
    rows = build_baseline_vs_simulated_rows(_simulation_result())
    assert [row["Intern ID"] for row in rows] == ["INT001", "INT002"]
    assert rows[0]["Baseline Final Score"] == 80.0
    assert rows[0]["Simulated Final Score"] == 85.0


def test_historical_trend_summary_extraction_returns_expected_counts() -> None:
    summary = extract_historical_trend_summary(
        {
            "trend_summary": {
                "improving_count": 1,
                "deteriorating_count": 2,
                "unchanged_count": 3,
                "new_in_period_count": 4,
                "missing_in_period_count": 5,
            }
        }
    )
    assert summary == {
        "improving_count": 1,
        "deteriorating_count": 2,
        "unchanged_count": 3,
        "new_in_period_count": 4,
        "missing_in_period_count": 5,
    }


def test_helper_functions_do_not_mutate_input_structures() -> None:
    simulation_result = _simulation_result()
    deltas = _deltas()
    before_simulation = copy.deepcopy(simulation_result)
    before_deltas = copy.deepcopy(deltas)

    build_baseline_vs_simulated_rows(simulation_result)
    build_delta_preview(deltas)
    group_changes_by_type(deltas["action_changes"])

    assert simulation_result == before_simulation
    assert deltas == before_deltas


def test_empty_delta_outputs_are_handled_explicitly() -> None:
    preview = build_delta_preview(
        {
            "metric_deltas": [],
            "category_changes": [],
            "driver_changes": [],
            "action_changes": [],
            "pattern_changes": [],
        }
    )
    assert preview == {
        "changed_final_scores": [],
        "category_transitions": [],
        "action_changes": [],
        "pattern_changes": [],
    }


def test_malformed_required_structures_fail_clearly() -> None:
    with pytest.raises(ValueError, match="missing required key 'baseline'"):
        build_baseline_vs_simulated_rows({"simulated": {"scores": {}}})

    with pytest.raises(ValueError, match="missing required key 'change_type'"):
        group_changes_by_type([{"action_key": "a1"}])
