from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"


@dataclass
class ScoreResults:
    task_metrics: pd.DataFrame
    summary: Dict[str, float | int | str | Dict[str, str]]


def load_csvs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the four core CSVs required for scoring.

    Returns:
        class_config, adjustment_config, tasks, flags
    """
    class_config = pd.read_csv(CONFIG_DIR / "class_config.csv")
    adjustment_config = pd.read_csv(CONFIG_DIR / "adjustment_config.csv")
    tasks = pd.read_csv(DATA_DIR / "tasks.csv")
    flags = pd.read_csv(DATA_DIR / "flags.csv")

    return class_config, adjustment_config, tasks, flags


def validate_inputs(
    class_config: pd.DataFrame,
    adjustment_config: pd.DataFrame,
    tasks: pd.DataFrame,
    flags: pd.DataFrame,
) -> None:
    """
    Basic schema and integrity checks for MVP safety.
    """
    required_class_cols = {
        "task_class",
        "class_name",
        "base_class_weight",
        "base_expected_hours",
        "is_active",
        "updated_at",
    }
    required_adjustment_cols = {
        "adjustment_code",
        "label",
        "multiplier_add",
        "min_bound",
        "max_bound",
        "is_active",
        "updated_at",
    }
    required_task_cols = {
        "task_id",
        "intern_id",
        "period",
        "task_class",
        "multi_state",
        "investments",
        "rental",
        "k1",
        "timer_start",
        "timer_end",
        "timer_running",
        "actual_time_hours",
        "minor_errors",
        "major_errors",
        "completed_at",
    }
    required_flag_cols = {"task_id", "flag_type", "flag_count"}

    missing_class = required_class_cols - set(class_config.columns)
    missing_adjustment = required_adjustment_cols - set(adjustment_config.columns)
    missing_task = required_task_cols - set(tasks.columns)
    missing_flags = required_flag_cols - set(flags.columns)

    if missing_class:
        raise ValueError(f"class_config.csv is missing columns: {sorted(missing_class)}")
    if missing_adjustment:
        raise ValueError(
            f"adjustment_config.csv is missing columns: {sorted(missing_adjustment)}"
        )
    if missing_task:
        raise ValueError(f"tasks.csv is missing columns: {sorted(missing_task)}")
    if missing_flags:
        raise ValueError(f"flags.csv is missing columns: {sorted(missing_flags)}")

    if class_config["task_class"].duplicated().any():
        raise ValueError("class_config.csv contains duplicate task_class values.")

    if adjustment_config["adjustment_code"].duplicated().any():
        raise ValueError("adjustment_config.csv contains duplicate adjustment_code values.")

    if tasks["task_id"].duplicated().any():
        raise ValueError("tasks.csv contains duplicate task_id values.")

    active_classes = set(class_config.loc[class_config["is_active"] == 1, "task_class"])
    unknown_classes = set(tasks["task_class"]) - active_classes
    if unknown_classes:
        raise ValueError(
            f"tasks.csv contains task_class values not active in class_config.csv: "
            f"{sorted(unknown_classes)}"
        )

    # Ensure all active adjustment codes exist as binary task columns.
    active_adjustments = adjustment_config.loc[
        adjustment_config["is_active"] == 1, "adjustment_code"
    ].tolist()

    missing_adjustment_columns = [col for col in active_adjustments if col not in tasks.columns]
    if missing_adjustment_columns:
        raise ValueError(
            f"tasks.csv is missing adjustment columns: {missing_adjustment_columns}"
        )

    # MVP sanity checks
    if (tasks["actual_time_hours"] <= 0).any():
        raise ValueError("All actual_time_hours values must be greater than 0.")

    if (tasks["minor_errors"] < 0).any() or (tasks["major_errors"] < 0).any():
        raise ValueError("Error counts cannot be negative.")

    for col in active_adjustments:
        invalid_values = set(tasks[col].dropna().unique()) - {0, 1}
        if invalid_values:
            raise ValueError(
                f"Adjustment column '{col}' contains non-binary values: {sorted(invalid_values)}"
            )

    normalized_task_ids = set(normalize_task_ids(tasks["task_id"]).dropna())
    normalized_flag_ids = set(normalize_task_ids(flags["task_id"]).dropna())
    unknown_flag_tasks = normalized_flag_ids - normalized_task_ids
    if unknown_flag_tasks:
        raise ValueError(
            f"flags.csv contains task_id values not present in tasks.csv: "
            f"{sorted(unknown_flag_tasks)}"
        )


def normalize_task_ids(task_ids: pd.Series) -> pd.Series:
    """
    Normalize task IDs like T01 and T001 to a comparable canonical form.
    """
    normalized = task_ids.astype(str).str.strip().str.upper()
    return normalized.str.replace(r"^([A-Z]+)0+(\d+)$", r"\1\2", regex=True)


def prepare_flags(flags: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce flag_count to numeric and replace invalid values with 0.
    """
    prepared_flags = flags.copy()
    prepared_flags["flag_count"] = pd.to_numeric(
        prepared_flags["flag_count"], errors="coerce"
    ).fillna(0)
    return prepared_flags


def get_active_adjustments(adjustment_config: pd.DataFrame) -> pd.DataFrame:
    """
    Return only active adjustments.
    """
    return adjustment_config.loc[adjustment_config["is_active"] == 1].copy()


def build_task_metrics(
    class_config: pd.DataFrame,
    adjustment_config: pd.DataFrame,
    tasks: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build row-level task metrics used for scoring and explainability.
    """
    active_classes = class_config.loc[class_config["is_active"] == 1].copy()
    active_adjustments = get_active_adjustments(adjustment_config)

    # Join class assumptions onto each task.
    task_metrics = tasks.merge(
        active_classes[["task_class", "class_name", "base_class_weight", "base_expected_hours"]],
        on="task_class",
        how="left",
        validate="many_to_one",
    )

    adjustment_codes = active_adjustments["adjustment_code"].tolist()
    adjustment_weights = dict(
        zip(active_adjustments["adjustment_code"], active_adjustments["multiplier_add"])
    )

    # Additive adjustment logic:
    # adjustment_multiplier = 1 + sum(active adjustment weights)
    task_metrics["adjustment_add"] = sum(
        task_metrics[col] * adjustment_weights[col] for col in adjustment_codes
    )
    task_metrics["adjustment_multiplier"] = 1 + task_metrics["adjustment_add"]

    # Derived row-level expectations and outputs
    task_metrics["expected_time_hours"] = (
        task_metrics["base_expected_hours"] * task_metrics["adjustment_multiplier"]
    )
    task_metrics["task_output"] = (
        task_metrics["base_class_weight"] * task_metrics["adjustment_multiplier"]
    )

    # Row-level efficiency is for display/diagnostics only
    task_metrics["efficiency_ratio_raw"] = (
        task_metrics["expected_time_hours"] / task_metrics["actual_time_hours"]
    )
    task_metrics["efficiency_ratio_capped"] = task_metrics["efficiency_ratio_raw"].clip(
        lower=0.5, upper=1.25
    )

    # Weighted review errors
    task_metrics["weighted_errors"] = (
        task_metrics["minor_errors"] * 0.5 + task_metrics["major_errors"] * 1.5
    )

    # Useful display field
    task_metrics["active_adjustments"] = task_metrics.apply(
        lambda row: ", ".join([code for code in adjustment_codes if row[code] == 1]) or "None",
        axis=1,
    )

    return task_metrics


def calculate_output_score(task_metrics: pd.DataFrame) -> float:
    """
    Output score = sum of per-task output values.
    """
    return float(task_metrics["task_output"].sum())


def calculate_efficiency_score(task_metrics: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Efficiency score = total expected time / total actual time, capped to [0.5, 1.25].

    Returns:
        total_expected_time, total_actual_time, efficiency_score
    """
    total_expected_time = float(task_metrics["expected_time_hours"].sum())
    total_actual_time = float(task_metrics["actual_time_hours"].sum())

    efficiency_raw = total_expected_time / total_actual_time
    efficiency_score = max(0.5, min(1.25, efficiency_raw))

    return total_expected_time, total_actual_time, efficiency_score


def calculate_accuracy_score(task_metrics: pd.DataFrame) -> Tuple[float, int]:
    """
    Accuracy score = 1 - (total weighted errors / total tasks)

    No lower cap is applied here because the MVP should transparently show if
    quality is genuinely poor. If needed later, this can be capped at 0.
    """
    total_weighted_errors = float(task_metrics["weighted_errors"].sum())
    total_tasks = int(len(task_metrics))

    accuracy_score = 1 - (total_weighted_errors / total_tasks)
    return accuracy_score, total_tasks


def calculate_contribution_modifier(
    positive_flags: int,
    negative_flags: int,
) -> float:
    """
    Contribution modifier = 1.0 + (0.015 * positive_flags) - (0.02 * negative_flags),
    capped to [0.9, 1.1].
    """
    contribution_raw = 1.0 + (0.015 * positive_flags) - (0.02 * negative_flags)
    return max(0.9, min(1.1, contribution_raw))


def calculate_flag_counts(flags: pd.DataFrame) -> Tuple[int, int]:
    """
    Aggregate positive and negative flag counts from flags.csv.
    """
    positive_flag_types = {"helped_peer", "proactive_update"}
    negative_flag_types = {"rework_requested", "blocked_escalated_late"}

    positive_flags = int(
        flags.loc[flags["flag_type"].isin(positive_flag_types), "flag_count"].sum()
    )
    negative_flags = int(
        flags.loc[flags["flag_type"].isin(negative_flag_types), "flag_count"].sum()
    )

    return positive_flags, negative_flags


def categorize_performance(performance_index: float) -> str:
    """
    Simple category thresholds for MVP.
    """
    if performance_index >= 0.90:
        return "Top"
    if performance_index >= 0.75:
        return "Solid"
    return "Risk"


def calculate_final_score(
    output_score: float,
    efficiency_score: float,
    accuracy_score: float,
    contribution_modifier: float,
) -> Tuple[float, float]:
    """
    Returns:
        final_score, performance_index

    performance_index isolates execution quality from volume:
        efficiency * accuracy * contribution
    """
    performance_index = efficiency_score * accuracy_score * contribution_modifier
    final_score = output_score * performance_index
    return final_score, performance_index


def build_diagnostics(
    efficiency_score: float,
    accuracy_score: float,
    contribution_modifier: float,
    positive_flags: int,
    negative_flags: int,
    performance_category: str,
) -> Dict[str, str]:
    """
    Deterministic diagnostic strings derived from computed scoring outputs.
    """
    if efficiency_score < 0.95:
        efficiency_diagnostic = (
            "Time management concern: actual completion time is materially above expected "
            "time."
        )
    elif efficiency_score <= 1.05:
        efficiency_diagnostic = "Time performance is near expected levels."
    else:
        efficiency_diagnostic = (
            "Strong time efficiency: actual completion time is better than expected."
        )

    if accuracy_score < 0.75:
        accuracy_diagnostic = "Quality risk: error rate is materially affecting performance."
    elif accuracy_score <= 0.90:
        accuracy_diagnostic = "Moderate quality performance with some error impact."
    else:
        accuracy_diagnostic = "Strong quality performance with low error impact."

    if positive_flags > negative_flags:
        contribution_diagnostic = (
            "Constructive contribution pattern: positive flags exceed negative flags."
        )
    elif negative_flags > positive_flags:
        contribution_diagnostic = (
            "Contribution concern: negative flags exceed positive flags."
        )
    else:
        contribution_diagnostic = (
            "Neutral contribution pattern: positive and negative flags are balanced."
        )

    weakest_metric = min(
        [
            ("efficiency_score", efficiency_score),
            ("accuracy_score", accuracy_score),
            ("contribution_modifier", contribution_modifier),
        ],
        key=lambda item: item[1],
    )[0]

    if performance_category == "Risk" and weakest_metric == "accuracy_score":
        overall_diagnostic = "Overall performance risk is primarily quality-driven."
    elif performance_category == "Risk" and weakest_metric == "efficiency_score":
        overall_diagnostic = "Overall performance risk is primarily speed-driven."
    elif performance_category == "Risk" and weakest_metric == "contribution_modifier":
        overall_diagnostic = (
            "Overall performance risk is primarily behavior/collaboration-driven."
        )
    else:
        overall_diagnostic = (
            "Overall performance reflects a mixed profile across output quality, speed, "
            "and contribution."
        )

    return {
        "efficiency_diagnostic": efficiency_diagnostic,
        "accuracy_diagnostic": accuracy_diagnostic,
        "contribution_diagnostic": contribution_diagnostic,
        "overall_diagnostic": overall_diagnostic,
    }


def run_scoring() -> ScoreResults:
    """
    Full scoring pipeline for the MVP.
    """
    class_config, adjustment_config, tasks, flags = load_csvs()
    validate_inputs(class_config, adjustment_config, tasks, flags)
    flags = prepare_flags(flags)

    task_metrics = build_task_metrics(class_config, adjustment_config, tasks)
    positive_flags, negative_flags = calculate_flag_counts(flags)

    output_score = calculate_output_score(task_metrics)
    total_expected_time, total_actual_time, efficiency_score = calculate_efficiency_score(
        task_metrics
    )
    accuracy_score, total_tasks = calculate_accuracy_score(task_metrics)
    total_weighted_errors = float(task_metrics["weighted_errors"].sum())
    contribution_modifier = calculate_contribution_modifier(
        positive_flags=positive_flags,
        negative_flags=negative_flags,
    )
    final_score, performance_index = calculate_final_score(
        output_score=output_score,
        efficiency_score=efficiency_score,
        accuracy_score=accuracy_score,
        contribution_modifier=contribution_modifier,
    )
    performance_category = categorize_performance(performance_index)
    diagnostics = build_diagnostics(
        efficiency_score=efficiency_score,
        accuracy_score=accuracy_score,
        contribution_modifier=contribution_modifier,
        positive_flags=positive_flags,
        negative_flags=negative_flags,
        performance_category=performance_category,
    )

    summary = {
        "intern_id": str(task_metrics["intern_id"].iloc[0]),
        "total_tasks": total_tasks,
        "output_score": round(output_score, 4),
        "total_expected_time": round(total_expected_time, 4),
        "total_actual_time": round(total_actual_time, 4),
        "efficiency_score": round(efficiency_score, 4),
        "total_weighted_errors": round(total_weighted_errors, 4),
        "accuracy_score": round(accuracy_score, 4),
        "contribution_modifier": round(contribution_modifier, 4),
        "performance_index": round(performance_index, 4),
        "final_score": round(final_score, 4),
        "performance_category": performance_category,
        "positive_flags": positive_flags,
        "negative_flags": negative_flags,
        "diagnostics": diagnostics,
    }

    return ScoreResults(task_metrics=task_metrics, summary=summary)


def print_summary(results: ScoreResults) -> None:
    """
    Terminal-friendly output for pre-UI testing.
    """
    summary = results.summary

    print(f"Output Score: {summary['output_score']}")
    print(f"Efficiency Score: {summary['efficiency_score']}")
    print(f"Accuracy Score: {summary['accuracy_score']}")
    print(f"Contribution Modifier: {summary['contribution_modifier']}")
    print(f"Final Score: {summary['final_score']}")
    print(f"Performance Index: {summary['performance_index']}")
    print(f"Category: {summary['performance_category']}")
    print(f"Positive flag count: {summary['positive_flags']}")
    print(f"Negative flag count: {summary['negative_flags']}")
    print(f"Efficiency Diagnostic: {summary['diagnostics']['efficiency_diagnostic']}")
    print(f"Accuracy Diagnostic: {summary['diagnostics']['accuracy_diagnostic']}")
    print(
        f"Contribution Diagnostic: "
        f"{summary['diagnostics']['contribution_diagnostic']}"
    )
    print(f"Overall Diagnostic: {summary['diagnostics']['overall_diagnostic']}")


if __name__ == "__main__":
    results = run_scoring()
    print_summary(results)
