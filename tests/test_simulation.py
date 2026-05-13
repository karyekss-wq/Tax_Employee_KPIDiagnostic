from __future__ import annotations

import pandas as pd
import pytest

from pipeline import PipelineInputs, load_pipeline_inputs, run_full_pipeline
from simulation import run_simulation


def _summary_snapshot(bundle: dict) -> dict:
    return {
        intern_id: dict(result.summary)
        for intern_id, result in bundle["scores"].items()
    }


def _baseline_inputs() -> PipelineInputs:
    return load_pipeline_inputs()


def test_simulation_does_not_mutate_baseline_config_or_data() -> None:
    inputs = _baseline_inputs()
    original_class_config = inputs.class_config.copy(deep=True)
    original_adjustment_config = inputs.adjustment_config.copy(deep=True)
    original_tasks = inputs.tasks.copy(deep=True)
    original_flags = inputs.flags.copy(deep=True)

    run_simulation(
        scenario_name="isolation check",
        class_expected_hours_overrides={"A": 2.5},
        class_weight_overrides={"B": 1.45},
        adjustment_multiplier_overrides={"multi_state": 0.25},
        baseline_inputs=inputs,
    )

    pd.testing.assert_frame_equal(inputs.class_config, original_class_config)
    pd.testing.assert_frame_equal(inputs.adjustment_config, original_adjustment_config)
    pd.testing.assert_frame_equal(inputs.tasks, original_tasks)
    pd.testing.assert_frame_equal(inputs.flags, original_flags)


def test_valid_class_expected_hours_override_returns_simulated_bundle() -> None:
    result = run_simulation(
        scenario_name="expected hours",
        class_expected_hours_overrides={"A": 2.5},
        baseline_inputs=_baseline_inputs(),
    )

    assert {"scenario_metadata", "baseline", "simulated"} <= set(result)
    assert result["scenario_metadata"]["override_count"] == 1
    assert _summary_snapshot(result["baseline"]) != _summary_snapshot(result["simulated"])


def test_valid_class_weight_override_returns_simulated_bundle() -> None:
    result = run_simulation(
        scenario_name="class weight",
        class_weight_overrides={"A": 1.1},
        baseline_inputs=_baseline_inputs(),
    )

    assert result["scenario_metadata"]["overrides_applied"]["class_weights"] == {"A": 1.1}
    assert _summary_snapshot(result["baseline"]) != _summary_snapshot(result["simulated"])


def test_valid_adjustment_multiplier_override_returns_simulated_bundle() -> None:
    result = run_simulation(
        scenario_name="adjustment multiplier",
        adjustment_multiplier_overrides={"multi_state": 0.25},
        baseline_inputs=_baseline_inputs(),
    )

    assert result["scenario_metadata"]["overrides_applied"]["adjustment_multipliers"] == {
        "multi_state": 0.25
    }
    assert _summary_snapshot(result["baseline"]) != _summary_snapshot(result["simulated"])


def test_unknown_task_class_override_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown task_class override"):
        run_simulation(
            scenario_name="bad class",
            class_expected_hours_overrides={"UNKNOWN": 2.5},
            baseline_inputs=_baseline_inputs(),
        )


def test_unknown_adjustment_override_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="Unknown adjustment override"):
        run_simulation(
            scenario_name="bad adjustment",
            adjustment_multiplier_overrides={"unknown_flag": 0.2},
            baseline_inputs=_baseline_inputs(),
        )


def test_nonnumeric_override_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="must be numeric"):
        run_simulation(
            scenario_name="bad numeric",
            class_weight_overrides={"A": "1.1"},
            baseline_inputs=_baseline_inputs(),
        )


def test_invalid_numeric_domain_fails_explicitly() -> None:
    with pytest.raises(ValueError, match="must be greater than 0"):
        run_simulation(
            scenario_name="bad domain",
            class_expected_hours_overrides={"A": 0},
            baseline_inputs=_baseline_inputs(),
        )


def test_simulated_run_returns_full_pipeline_bundles() -> None:
    result = run_simulation(
        scenario_name="bundle shape",
        class_weight_overrides={"B": 1.4},
        baseline_inputs=_baseline_inputs(),
    )

    for key in ["baseline", "simulated"]:
        assert {
            "scores",
            "diagnostics",
            "normalized_insights",
            "system_patterns",
            "manager_actions",
        } <= set(result[key])


def test_simulation_reuses_pipeline_runner_for_baseline_and_simulated_runs() -> None:
    inputs = _baseline_inputs()
    calls = []

    def spy_runner(**kwargs):
        calls.append(
            {
                "class_config": kwargs["class_config"].copy(deep=True),
                "adjustment_config": kwargs["adjustment_config"].copy(deep=True),
            }
        )
        return {
            "scores": {},
            "diagnostics": {},
            "normalized_insights": [],
            "system_patterns": {"system_patterns": [], "pattern_summary": {}},
            "manager_actions": {"intern_actions": [], "team_actions": [], "action_summary": {}},
        }

    run_simulation(
        scenario_name="pipeline spy",
        class_weight_overrides={"A": 1.1},
        baseline_inputs=inputs,
        pipeline_runner=spy_runner,
    )

    assert len(calls) == 2
    baseline_weight = calls[0]["class_config"].loc[
        calls[0]["class_config"]["task_class"] == "A", "base_class_weight"
    ].iloc[0]
    simulated_weight = calls[1]["class_config"].loc[
        calls[1]["class_config"]["task_class"] == "A", "base_class_weight"
    ].iloc[0]
    assert baseline_weight == 1.0
    assert simulated_weight == 1.1


def test_baseline_run_remains_identical_before_and_after_simulation() -> None:
    inputs = _baseline_inputs()
    before = _summary_snapshot(
        run_full_pipeline(
            class_config=inputs.class_config.copy(deep=True),
            adjustment_config=inputs.adjustment_config.copy(deep=True),
            tasks=inputs.tasks.copy(deep=True),
            flags=inputs.flags.copy(deep=True),
        )
    )

    run_simulation(
        scenario_name="baseline stability",
        class_expected_hours_overrides={"A": 2.5},
        class_weight_overrides={"B": 1.45},
        adjustment_multiplier_overrides={"multi_state": 0.25},
        baseline_inputs=inputs,
    )

    after = _summary_snapshot(
        run_full_pipeline(
            class_config=inputs.class_config.copy(deep=True),
            adjustment_config=inputs.adjustment_config.copy(deep=True),
            tasks=inputs.tasks.copy(deep=True),
            flags=inputs.flags.copy(deep=True),
        )
    )
    assert before == after
