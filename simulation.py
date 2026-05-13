from __future__ import annotations

from copy import deepcopy
from numbers import Real
from typing import Any, Callable

import pandas as pd

from pipeline import PipelineInputs, load_pipeline_inputs, run_full_pipeline


PipelineRunner = Callable[..., dict[str, Any]]


def _copy_inputs(inputs: PipelineInputs) -> PipelineInputs:
    return PipelineInputs(
        class_config=inputs.class_config.copy(deep=True),
        adjustment_config=inputs.adjustment_config.copy(deep=True),
        tasks=inputs.tasks.copy(deep=True),
        flags=inputs.flags.copy(deep=True),
    )


def _require_numeric(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric.")
    return float(value)


def _validate_override_mapping(overrides: dict[str, Any] | None, label: str) -> dict[str, Any]:
    if overrides is None:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{label} overrides must be provided as a dict.")
    return deepcopy(overrides)


def _apply_class_overrides(
    class_config: pd.DataFrame,
    *,
    class_expected_hours_overrides: dict[str, Any],
    class_weight_overrides: dict[str, Any],
) -> None:
    class_ids = set(class_config["task_class"].astype(str))

    for task_class, raw_value in class_expected_hours_overrides.items():
        if task_class not in class_ids:
            raise ValueError(f"Unknown task_class override for expected hours: {task_class}")
        value = _require_numeric(raw_value, f"Expected-hours override for task_class '{task_class}'")
        if value <= 0:
            raise ValueError(
                f"Expected-hours override for task_class '{task_class}' must be greater than 0."
            )
        class_config.loc[
            class_config["task_class"] == task_class, "base_expected_hours"
        ] = value

    for task_class, raw_value in class_weight_overrides.items():
        if task_class not in class_ids:
            raise ValueError(f"Unknown task_class override for class weight: {task_class}")
        value = _require_numeric(raw_value, f"Class-weight override for task_class '{task_class}'")
        if value <= 0:
            raise ValueError(
                f"Class-weight override for task_class '{task_class}' must be greater than 0."
            )
        class_config.loc[class_config["task_class"] == task_class, "base_class_weight"] = value


def _apply_adjustment_overrides(
    adjustment_config: pd.DataFrame,
    *,
    adjustment_multiplier_overrides: dict[str, Any],
) -> None:
    adjustment_ids = set(adjustment_config["adjustment_code"].astype(str))

    for adjustment_code, raw_value in adjustment_multiplier_overrides.items():
        if adjustment_code not in adjustment_ids:
            raise ValueError(f"Unknown adjustment override: {adjustment_code}")
        value = _require_numeric(
            raw_value,
            f"Adjustment multiplier override for adjustment_code '{adjustment_code}'",
        )
        row = adjustment_config.loc[adjustment_config["adjustment_code"] == adjustment_code].iloc[0]
        min_bound = float(row["min_bound"])
        max_bound = float(row["max_bound"])
        if value < min_bound or value > max_bound:
            raise ValueError(
                f"Adjustment multiplier override for adjustment_code '{adjustment_code}' "
                f"must be between {min_bound} and {max_bound}."
            )
        adjustment_config.loc[
            adjustment_config["adjustment_code"] == adjustment_code, "multiplier_add"
        ] = value


def _run_pipeline(pipeline_runner: PipelineRunner, inputs: PipelineInputs) -> dict[str, Any]:
    return pipeline_runner(
        class_config=inputs.class_config,
        adjustment_config=inputs.adjustment_config,
        tasks=inputs.tasks,
        flags=inputs.flags,
    )


def run_simulation(
    *,
    scenario_name: str,
    class_expected_hours_overrides: dict[str, Any] | None = None,
    class_weight_overrides: dict[str, Any] | None = None,
    adjustment_multiplier_overrides: dict[str, Any] | None = None,
    baseline_inputs: PipelineInputs | None = None,
    pipeline_runner: PipelineRunner = run_full_pipeline,
) -> dict[str, Any]:
    """
    Run a sandboxed what-if scenario against cloned config/data inputs.
    """
    scenario = str(scenario_name)
    inputs = baseline_inputs if baseline_inputs is not None else load_pipeline_inputs()

    expected_hours = _validate_override_mapping(
        class_expected_hours_overrides, "class expected-hours"
    )
    class_weights = _validate_override_mapping(class_weight_overrides, "class weight")
    adjustment_multipliers = _validate_override_mapping(
        adjustment_multiplier_overrides, "adjustment multiplier"
    )

    baseline_run_inputs = _copy_inputs(inputs)
    baseline_bundle = _run_pipeline(pipeline_runner, baseline_run_inputs)

    simulated_inputs = _copy_inputs(inputs)
    _apply_class_overrides(
        simulated_inputs.class_config,
        class_expected_hours_overrides=expected_hours,
        class_weight_overrides=class_weights,
    )
    _apply_adjustment_overrides(
        simulated_inputs.adjustment_config,
        adjustment_multiplier_overrides=adjustment_multipliers,
    )
    simulated_bundle = _run_pipeline(pipeline_runner, simulated_inputs)

    overrides_applied = {
        "class_expected_hours": expected_hours,
        "class_weights": class_weights,
        "adjustment_multipliers": adjustment_multipliers,
    }

    return {
        "scenario_metadata": {
            "scenario_name": scenario,
            "overrides_applied": overrides_applied,
            "override_count": sum(len(values) for values in overrides_applied.values()),
        },
        "baseline": baseline_bundle,
        "simulated": simulated_bundle,
    }
