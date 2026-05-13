from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import simulation
from delta_analysis import build_simulation_deltas


METRIC_DEFAULTS = {
    "final_score": 80.0,
    "performance_index": 0.8,
    "output_score": 100.0,
    "efficiency_score": 1.0,
    "accuracy_score": 0.9,
    "contribution_modifier": 1.0,
    "performance_category": "Solid",
}


def _score(**overrides):
    summary = dict(METRIC_DEFAULTS)
    summary.update(overrides)
    return SimpleNamespace(summary=summary)


def _diagnostic(
    *,
    strength: str = "output_score",
    weakness: str = "accuracy_score",
    final_driver: str = "accuracy_score",
):
    return {
        "normalized_insights": [
            {
                "insight_key": "primary_strength_driver",
                "metric_source": strength,
            },
            {
                "insight_key": "primary_weakness_driver",
                "metric_source": weakness,
            },
            {
                "insight_key": "dominant_final_score_driver",
                "metric_source": final_driver,
            },
        ]
    }


def _action(
    action_key: str,
    *,
    priority: str = "low",
    intern_id: str = "INT001",
    action_type: str = "watchlist",
):
    return {
        "action_key": action_key,
        "action_type": action_type,
        "target_scope": "intern",
        "target_id": intern_id,
        "priority_level": priority,
        "intern_id": intern_id,
    }


def _team_action(action_key: str, *, priority: str = "moderate"):
    return {
        "action_key": action_key,
        "action_type": "training_focus",
        "target_scope": "team",
        "target_id": "accuracy_score",
        "priority_level": priority,
    }


def _pattern(
    pattern_key: str,
    *,
    scope: str = "isolated",
    frequency: float = 0.25,
    pattern_type: str = "recurring_weakness",
):
    return {
        "pattern_key": pattern_key,
        "pattern_type": pattern_type,
        "scope_classification": scope,
        "frequency": frequency,
    }


def _bundle(
    *,
    score_overrides: dict | None = None,
    diagnostic_overrides: dict | None = None,
    intern_actions: list[dict] | None = None,
    team_actions: list[dict] | None = None,
    patterns: list[dict] | None = None,
):
    diagnostic_args = diagnostic_overrides or {}
    return {
        "scores": {"INT001": _score(**(score_overrides or {}))},
        "diagnostics": {"INT001": _diagnostic(**diagnostic_args)},
        "normalized_insights": [],
        "manager_actions": {
            "intern_actions": intern_actions if intern_actions is not None else [_action("a1")],
            "team_actions": team_actions if team_actions is not None else [],
            "action_summary": {},
        },
        "system_patterns": {
            "system_patterns": patterns if patterns is not None else [_pattern("p1")],
            "pattern_summary": {},
        },
    }


def _find_metric(deltas: dict, metric_name: str) -> dict:
    return next(row for row in deltas["metric_deltas"] if row["metric_name"] == metric_name)


def test_metric_delta_increase() -> None:
    deltas = build_simulation_deltas(
        _bundle(score_overrides={"final_score": 80.0}),
        _bundle(score_overrides={"final_score": 85.0}),
    )
    row = _find_metric(deltas, "final_score")
    assert row["absolute_delta"] == 5.0
    assert row["direction"] == "increase"


def test_metric_delta_decrease() -> None:
    deltas = build_simulation_deltas(
        _bundle(score_overrides={"final_score": 80.0}),
        _bundle(score_overrides={"final_score": 75.0}),
    )
    row = _find_metric(deltas, "final_score")
    assert row["absolute_delta"] == -5.0
    assert row["direction"] == "decrease"


def test_metric_delta_no_change() -> None:
    deltas = build_simulation_deltas(
        _bundle(score_overrides={"final_score": 80.0}),
        _bundle(score_overrides={"final_score": 80.0}),
    )
    assert _find_metric(deltas, "final_score")["direction"] == "no_change"


def test_category_transition_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(score_overrides={"performance_category": "Risk"}),
        _bundle(score_overrides={"performance_category": "Solid"}),
    )
    assert deltas["category_changes"] == [
        {
            "intern_id": "INT001",
            "baseline_category": "Risk",
            "simulated_category": "Solid",
            "changed": True,
        }
    ]


def test_unchanged_category_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(score_overrides={"performance_category": "Solid"}),
        _bundle(score_overrides={"performance_category": "Solid"}),
    )
    assert deltas["category_changes"][0]["changed"] is False


def test_driver_change_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(diagnostic_overrides={"weakness": "accuracy_score"}),
        _bundle(diagnostic_overrides={"weakness": "efficiency_score"}),
    )
    row = next(
        item
        for item in deltas["driver_changes"]
        if item["driver_type"] == "primary_weakness_driver"
    )
    assert row["baseline_driver"] == "accuracy_score"
    assert row["simulated_driver"] == "efficiency_score"
    assert row["changed"] is True


def test_unchanged_driver_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(diagnostic_overrides={"strength": "output_score"}),
        _bundle(diagnostic_overrides={"strength": "output_score"}),
    )
    row = next(
        item
        for item in deltas["driver_changes"]
        if item["driver_type"] == "primary_strength_driver"
    )
    assert row["changed"] is False


def test_action_added_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(intern_actions=[]),
        _bundle(intern_actions=[_action("a1")]),
    )
    assert deltas["action_changes"][0]["change_type"] == "added"


def test_action_removed_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(intern_actions=[_action("a1")]),
        _bundle(intern_actions=[]),
    )
    assert deltas["action_changes"][0]["change_type"] == "removed"


def test_action_priority_change_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(intern_actions=[_action("a1", priority="low")]),
        _bundle(intern_actions=[_action("a1", priority="high")]),
    )
    row = deltas["action_changes"][0]
    assert row["change_type"] == "priority_changed"
    assert row["baseline_priority"] == "low"
    assert row["simulated_priority"] == "high"


def test_pattern_scope_change_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(patterns=[_pattern("p1", scope="systemic", frequency=0.5)]),
        _bundle(patterns=[_pattern("p1", scope="emerging", frequency=0.5)]),
    )
    row = deltas["pattern_changes"][0]
    assert row["change_type"] == "scope_changed"
    assert row["baseline_scope"] == "systemic"
    assert row["simulated_scope"] == "emerging"


def test_resolved_pattern_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(patterns=[_pattern("p1")]),
        _bundle(patterns=[]),
    )
    assert deltas["pattern_changes"][0]["change_type"] == "resolved"


def test_new_pattern_detection() -> None:
    deltas = build_simulation_deltas(
        _bundle(patterns=[]),
        _bundle(patterns=[_pattern("p1")]),
    )
    assert deltas["pattern_changes"][0]["change_type"] == "introduced"


def test_deterministic_output_consistency_with_ordering_differences() -> None:
    baseline = _bundle(
        intern_actions=[_action("b"), _action("a")],
        team_actions=[_team_action("team")],
        patterns=[_pattern("p2"), _pattern("p1")],
    )
    simulated = _bundle(
        intern_actions=[_action("a"), _action("b")],
        team_actions=[_team_action("team")],
        patterns=[_pattern("p1"), _pattern("p2")],
    )

    first = build_simulation_deltas(baseline, simulated)
    second = build_simulation_deltas(baseline, simulated)
    assert first == second
    assert [row["action_key"] for row in first["action_changes"]] == ["a", "b", "team"]
    assert [row["pattern_key"] for row in first["pattern_changes"]] == ["p1", "p2"]


def test_delta_layer_does_not_mutate_inputs() -> None:
    baseline = _bundle()
    simulated = _bundle(score_overrides={"final_score": 85.0})
    baseline_before = copy.deepcopy(baseline)
    simulated_before = copy.deepcopy(simulated)

    build_simulation_deltas(baseline, simulated)

    assert baseline == baseline_before
    assert simulated == simulated_before


def test_delta_layer_does_not_rerun_simulation_logic(monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("simulation must not be rerun")

    monkeypatch.setattr(simulation, "run_simulation", fail_if_called)

    deltas = build_simulation_deltas(_bundle(), _bundle())
    assert deltas["metric_deltas"]


def test_equal_baseline_and_simulated_bundles_produce_stable_no_change_outputs() -> None:
    baseline = _bundle(
        intern_actions=[_action("a1", priority="low")],
        patterns=[_pattern("p1", scope="isolated", frequency=0.25)],
    )
    simulated = copy.deepcopy(baseline)

    first = build_simulation_deltas(baseline, simulated)
    second = build_simulation_deltas(baseline, simulated)

    assert first == second
    assert all(row["direction"] == "no_change" for row in first["metric_deltas"])
    assert all(not row["changed"] for row in first["category_changes"])
    assert all(not row["changed"] for row in first["driver_changes"])
    assert all(row["change_type"] == "unchanged" for row in first["action_changes"])
    assert all(row["change_type"] == "unchanged" for row in first["pattern_changes"])


def test_malformed_bundle_fails_clearly() -> None:
    with pytest.raises(ValueError, match="missing required key 'scores'"):
        build_simulation_deltas({}, _bundle())
