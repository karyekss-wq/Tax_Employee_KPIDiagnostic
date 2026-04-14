from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"


@dataclass
class ScoreResults:
    task_metrics: pd.DataFrame
    summary: Dict[str, float | int | str | Dict[str, str]]
    attribution: Dict[str, Any]


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

    numeric_flag_counts = pd.to_numeric(flags["flag_count"], errors="raise")
    if (numeric_flag_counts < 0).any():
        raise ValueError("flag_count values cannot be negative.")

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
    Coerce flag_count to numeric after validation has rejected malformed values.
    """
    prepared_flags = flags.copy()
    prepared_flags["flag_count"] = pd.to_numeric(prepared_flags["flag_count"], errors="raise")
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


def validate_attribution_columns(df: pd.DataFrame, required_cols: Iterable[str], name: str) -> None:
    """
    Fail clearly if an attribution helper is called without its required columns.
    """
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"{name} is missing attribution columns: {sorted(missing_cols)}")


def records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a dataframe to plain records without mutating caller-owned data.
    """
    return df.to_dict(orient="records")


def build_output_attribution(
    task_metrics: pd.DataFrame,
    adjustment_config: pd.DataFrame,
    output_score: float,
) -> Dict[str, Any]:
    """
    Explain output score by task, task class, and active adjustment code.
    """
    validate_attribution_columns(
        task_metrics,
        [
            "task_id",
            "task_class",
            "class_name",
            "base_class_weight",
            "adjustment_multiplier",
            "task_output",
        ],
        "task_metrics",
    )
    active_adjustments = get_active_adjustments(adjustment_config)
    adjustment_codes = active_adjustments["adjustment_code"].tolist()
    validate_attribution_columns(task_metrics, adjustment_codes, "task_metrics")

    tasks = task_metrics.copy()
    tasks["output_share"] = (
        tasks["task_output"] / output_score if output_score else 0.0
    )

    by_task = tasks[
        [
            "task_id",
            "task_class",
            "class_name",
            "base_class_weight",
            "adjustment_multiplier",
            "task_output",
            "output_share",
        ]
    ].sort_values(["task_output", "task_id"], ascending=[False, True], kind="mergesort")

    by_class = (
        tasks.groupby(["task_class", "class_name"], as_index=False, sort=False)
        .agg(
            task_count=("task_id", "count"),
            output_contribution=("task_output", "sum"),
        )
    )
    by_class["output_share"] = (
        by_class["output_contribution"] / output_score if output_score else 0.0
    )
    by_class = by_class.sort_values(
        ["output_contribution", "task_class"], ascending=[False, True], kind="mergesort"
    )

    adjustment_rows = []
    for row in active_adjustments.sort_values("adjustment_code", kind="mergesort").itertuples(
        index=False
    ):
        flagged = tasks[row.adjustment_code] == 1
        output_effect = float(
            (tasks.loc[flagged, "base_class_weight"] * row.multiplier_add).sum()
        )
        adjustment_rows.append(
            {
                "adjustment_code": row.adjustment_code,
                "label": row.label,
                "task_count": int(flagged.sum()),
                "multiplier_add": float(row.multiplier_add),
                "output_effect": output_effect,
                "output_effect_share": output_effect / output_score if output_score else 0.0,
            }
        )

    base_output_without_adjustments = float(tasks["base_class_weight"].sum())
    adjustment_output_effect = float(
        sum(row["output_effect"] for row in adjustment_rows)
    )

    return {
        "by_task": records_from_df(by_task),
        "by_class": records_from_df(by_class),
        "by_adjustment": adjustment_rows,
        "reconciliation": {
            "base_output_without_adjustments": base_output_without_adjustments,
            "adjustment_output_effect": adjustment_output_effect,
            "total_output": float(output_score),
        },
    }


def build_efficiency_attribution(task_metrics: pd.DataFrame) -> Dict[str, Any]:
    """
    Explain efficiency by task-level expected hours, actual hours, and deltas.
    """
    validate_attribution_columns(
        task_metrics,
        ["task_id", "task_class", "expected_time_hours", "actual_time_hours"],
        "task_metrics",
    )

    tasks = task_metrics[
        ["task_id", "task_class", "expected_time_hours", "actual_time_hours"]
    ].copy()
    tasks["time_delta_hours"] = tasks["actual_time_hours"] - tasks["expected_time_hours"]
    tasks["overrun_hours"] = tasks["time_delta_hours"].clip(lower=0)
    tasks["underrun_hours"] = (-tasks["time_delta_hours"]).clip(lower=0)

    total_overrun_hours = float(tasks["overrun_hours"].sum())
    tasks["inefficiency_share"] = (
        tasks["overrun_hours"] / total_overrun_hours if total_overrun_hours else 0.0
    )

    by_task = tasks.sort_values(
        ["time_delta_hours", "task_id"], ascending=[False, True], kind="mergesort"
    )
    largest_overruns = by_task[by_task["time_delta_hours"] > 0].head(5)
    largest_underruns = tasks[tasks["time_delta_hours"] < 0].sort_values(
        ["time_delta_hours", "task_id"], ascending=[True, True], kind="mergesort"
    ).head(5)

    return {
        "by_task": records_from_df(by_task),
        "largest_overruns": records_from_df(largest_overruns),
        "largest_underruns": records_from_df(largest_underruns),
        "reconciliation": {
            "total_expected_time": float(tasks["expected_time_hours"].sum()),
            "total_actual_time": float(tasks["actual_time_hours"].sum()),
            "total_time_delta_hours": float(tasks["time_delta_hours"].sum()),
            "total_overrun_hours": total_overrun_hours,
        },
    }


def build_accuracy_attribution(task_metrics: pd.DataFrame) -> Dict[str, Any]:
    """
    Explain accuracy loss by task and by existing error severity fields.
    """
    validate_attribution_columns(
        task_metrics,
        ["task_id", "task_class", "minor_errors", "major_errors", "weighted_errors"],
        "task_metrics",
    )

    tasks = task_metrics[
        ["task_id", "task_class", "minor_errors", "major_errors", "weighted_errors"]
    ].copy()
    tasks["minor_error_impact"] = tasks["minor_errors"] * 0.5
    tasks["major_error_impact"] = tasks["major_errors"] * 1.5
    tasks["total_error_count"] = tasks["minor_errors"] + tasks["major_errors"]

    total_weighted_errors = float(tasks["weighted_errors"].sum())
    total_tasks = int(len(tasks))
    tasks["accuracy_loss_share"] = (
        tasks["weighted_errors"] / total_weighted_errors if total_weighted_errors else 0.0
    )
    tasks["accuracy_score_impact"] = (
        tasks["weighted_errors"] / total_tasks if total_tasks else 0.0
    )

    by_task = tasks.sort_values(
        ["weighted_errors", "task_id"], ascending=[False, True], kind="mergesort"
    )
    top_error_drivers = by_task[by_task["weighted_errors"] > 0].head(5)

    severity_breakdown = [
        {
            "severity": "minor",
            "error_count": int(tasks["minor_errors"].sum()),
            "weighted_error_impact": float(tasks["minor_error_impact"].sum()),
            "accuracy_score_impact": (
                float(tasks["minor_error_impact"].sum()) / total_tasks
                if total_tasks
                else 0.0
            ),
        },
        {
            "severity": "major",
            "error_count": int(tasks["major_errors"].sum()),
            "weighted_error_impact": float(tasks["major_error_impact"].sum()),
            "accuracy_score_impact": (
                float(tasks["major_error_impact"].sum()) / total_tasks
                if total_tasks
                else 0.0
            ),
        },
    ]

    return {
        "by_task": records_from_df(by_task),
        "top_error_drivers": records_from_df(top_error_drivers),
        "by_severity": severity_breakdown,
        "reconciliation": {
            "total_weighted_errors": total_weighted_errors,
            "total_tasks": total_tasks,
            "accuracy_loss": total_weighted_errors / total_tasks if total_tasks else 0.0,
        },
    }


def build_contribution_attribution(
    flags: pd.DataFrame,
    positive_flags: int,
    negative_flags: int,
    contribution_modifier: float,
) -> Dict[str, Any]:
    """
    Explain contribution modifier by flag type using the locked formula.
    """
    validate_attribution_columns(flags, ["flag_type", "flag_count"], "flags")
    positive_flag_types = {"helped_peer", "proactive_update"}
    negative_flag_types = {"rework_requested", "blocked_escalated_late"}

    grouped = (
        flags.copy()
        .groupby("flag_type", as_index=False, sort=False)
        .agg(flag_count=("flag_count", "sum"))
    )

    positive_by_type = grouped[grouped["flag_type"].isin(positive_flag_types)].copy()
    positive_by_type["modifier_effect"] = positive_by_type["flag_count"] * 0.015
    positive_by_type = positive_by_type.sort_values(
        ["modifier_effect", "flag_type"], ascending=[False, True], kind="mergesort"
    )

    negative_by_type = grouped[grouped["flag_type"].isin(negative_flag_types)].copy()
    negative_by_type["modifier_effect"] = negative_by_type["flag_count"] * -0.02
    negative_by_type["absolute_modifier_effect"] = negative_by_type[
        "modifier_effect"
    ].abs()
    negative_by_type = negative_by_type.sort_values(
        ["absolute_modifier_effect", "flag_type"],
        ascending=[False, True],
        kind="mergesort",
    ).drop(columns=["absolute_modifier_effect"])

    raw_positive_effect = 0.015 * positive_flags
    raw_negative_effect = -0.02 * negative_flags
    raw_modifier_before_cap = 1.0 + raw_positive_effect + raw_negative_effect

    return {
        "positive_by_type": records_from_df(positive_by_type),
        "negative_by_type": records_from_df(negative_by_type),
        "raw_positive_effect": float(raw_positive_effect),
        "raw_negative_effect": float(raw_negative_effect),
        "raw_modifier_before_cap": float(raw_modifier_before_cap),
        "final_modifier_after_cap": float(contribution_modifier),
        "cap_applied": bool(raw_modifier_before_cap != contribution_modifier),
        "reconciliation": {
            "positive_flags": int(positive_flags),
            "negative_flags": int(negative_flags),
        },
    }


def build_overall_attribution(
    output_attribution: Dict[str, Any],
    efficiency_attribution: Dict[str, Any],
    accuracy_attribution: Dict[str, Any],
    contribution_attribution: Dict[str, Any],
    summary_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build deterministic plain-language attribution from computed values only.
    """
    by_task_output = output_attribution["by_task"]
    overruns = efficiency_attribution["largest_overruns"]
    underruns = efficiency_attribution["largest_underruns"]
    error_drivers = accuracy_attribution["top_error_drivers"]

    top_output_task = by_task_output[0] if by_task_output else None
    top_underrun_task = underruns[0] if underruns else None
    top_overrun_task = overruns[0] if overruns else None
    top_error_task = error_drivers[0] if error_drivers else None

    positive_candidates = []
    if top_output_task:
        positive_candidates.append(
            {
                "driver": "highest_output_task",
                "label": top_output_task["task_id"],
                "value": float(top_output_task["task_output"]),
                "detail": (
                    f"{top_output_task['task_id']} contributed "
                    f"{top_output_task['task_output']} output points."
                ),
            }
        )
    if top_underrun_task:
        positive_candidates.append(
            {
                "driver": "largest_time_underrun",
                "label": top_underrun_task["task_id"],
                "value": float(top_underrun_task["underrun_hours"]),
                "detail": (
                    f"{top_underrun_task['task_id']} finished "
                    f"{top_underrun_task['underrun_hours']} hours under expected time."
                ),
            }
        )
    if contribution_attribution["raw_positive_effect"] > 0:
        positive_candidates.append(
            {
                "driver": "positive_flags",
                "label": "positive flags",
                "value": float(contribution_attribution["raw_positive_effect"]),
                "detail": (
                    f"Positive flags increased the raw contribution modifier by "
                    f"{contribution_attribution['raw_positive_effect']}."
                ),
            }
        )

    negative_candidates = []
    if top_error_task:
        negative_candidates.append(
            {
                "driver": "largest_error_driver",
                "label": top_error_task["task_id"],
                "value": float(top_error_task["weighted_errors"]),
                "detail": (
                    f"{top_error_task['task_id']} added "
                    f"{top_error_task['weighted_errors']} weighted errors."
                ),
            }
        )
    if top_overrun_task:
        negative_candidates.append(
            {
                "driver": "largest_time_overrun",
                "label": top_overrun_task["task_id"],
                "value": float(top_overrun_task["overrun_hours"]),
                "detail": (
                    f"{top_overrun_task['task_id']} ran "
                    f"{top_overrun_task['overrun_hours']} hours over expected time."
                ),
            }
        )
    if contribution_attribution["raw_negative_effect"] < 0:
        negative_candidates.append(
            {
                "driver": "negative_flags",
                "label": "negative flags",
                "value": abs(float(contribution_attribution["raw_negative_effect"])),
                "detail": (
                    f"Negative flags decreased the raw contribution modifier by "
                    f"{abs(contribution_attribution['raw_negative_effect'])}."
                ),
            }
        )

    top_positive_driver = max(
        positive_candidates, key=lambda item: item["value"]
    ) if positive_candidates else None
    top_negative_driver = max(
        negative_candidates, key=lambda item: item["value"]
    ) if negative_candidates else None

    weakest_component = min(
        [
            ("efficiency_score", summary_values["efficiency_score"]),
            ("accuracy_score", summary_values["accuracy_score"]),
            ("contribution_modifier", summary_values["contribution_modifier"]),
        ],
        key=lambda item: item[1],
    )

    summary_lines = []
    if top_positive_driver:
        summary_lines.append(f"Most improved: {top_positive_driver['detail']}")
    if top_negative_driver:
        summary_lines.append(f"Most harmed: {top_negative_driver['detail']}")
    summary_lines.append(
        f"Category influence: {weakest_component[0]} is the lowest component at "
        f"{weakest_component[1]}."
    )
    summary_lines.append(
        f"Final score reconciliation: {summary_values['output_score']} output x "
        f"{summary_values['performance_index']} performance index = "
        f"{summary_values['final_score']}."
    )

    return {
        "top_positive_driver": top_positive_driver,
        "top_negative_driver": top_negative_driver,
        "category_driver": {
            "component": weakest_component[0],
            "value": float(weakest_component[1]),
            "category": summary_values["performance_category"],
        },
        "summary_lines": summary_lines,
    }


def build_attribution_payload(
    task_metrics: pd.DataFrame,
    adjustment_config: pd.DataFrame,
    flags: pd.DataFrame,
    summary_values: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build all deterministic attribution outputs without changing score calculations.
    """
    output_attribution = build_output_attribution(
        task_metrics=task_metrics,
        adjustment_config=adjustment_config,
        output_score=summary_values["output_score_raw"],
    )
    efficiency_attribution = build_efficiency_attribution(task_metrics)
    accuracy_attribution = build_accuracy_attribution(task_metrics)
    contribution_attribution = build_contribution_attribution(
        flags=flags,
        positive_flags=summary_values["positive_flags"],
        negative_flags=summary_values["negative_flags"],
        contribution_modifier=summary_values["contribution_modifier_raw"],
    )
    overall_attribution = build_overall_attribution(
        output_attribution=output_attribution,
        efficiency_attribution=efficiency_attribution,
        accuracy_attribution=accuracy_attribution,
        contribution_attribution=contribution_attribution,
        summary_values=summary_values,
    )

    return {
        "output_attribution": output_attribution,
        "efficiency_attribution": efficiency_attribution,
        "accuracy_attribution": accuracy_attribution,
        "contribution_attribution": contribution_attribution,
        "overall_attribution": overall_attribution,
    }


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

    summary_values = {
        "output_score_raw": output_score,
        "output_score": output_score,
        "efficiency_score": efficiency_score,
        "accuracy_score": accuracy_score,
        "contribution_modifier_raw": contribution_modifier,
        "contribution_modifier": contribution_modifier,
        "performance_index": performance_index,
        "final_score": final_score,
        "performance_category": performance_category,
        "positive_flags": positive_flags,
        "negative_flags": negative_flags,
    }
    attribution = build_attribution_payload(
        task_metrics=task_metrics,
        adjustment_config=adjustment_config,
        flags=flags,
        summary_values=summary_values,
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

    return ScoreResults(task_metrics=task_metrics, summary=summary, attribution=attribution)


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
