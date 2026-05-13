from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from cross_intern_patterns import build_cross_intern_patterns
from delta_analysis import build_simulation_deltas
from diagnostic_insights import build_diagnostic_insights
from manager_actions import build_manager_actions
from scenario_state import (
    delete_scenario,
    list_scenarios,
    load_scenario,
    run_saved_scenario,
    save_scenario,
)
from scoring import ScoreResults, run_scoring
from simulation import run_simulation


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


st.set_page_config(
    page_title="Tax Intern KPI Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_results() -> dict[str, ScoreResults]:
    """
    Load batch-scored intern results once per session unless files change.
    """
    return run_scoring()


def load_class_config() -> pd.DataFrame:
    """
    Load the full class config from disk.
    """
    return pd.read_csv(CONFIG_DIR / "class_config.csv")


def load_adjustment_config() -> pd.DataFrame:
    """
    Load the full adjustment config from disk.
    """
    return pd.read_csv(CONFIG_DIR / "adjustment_config.csv")


def build_class_config_editor_df(class_config: pd.DataFrame) -> pd.DataFrame:
    """
    Build the admin editor view for class config with governance context visible.
    """
    return class_config[
        [
            "task_class",
            "class_name",
            "base_class_weight",
            "base_expected_hours",
            "is_active",
            "updated_at",
        ]
    ].copy()


def build_adjustment_config_editor_df(adjustment_config: pd.DataFrame) -> pd.DataFrame:
    """
    Build the admin editor view for adjustment config with governance context visible.
    """
    return adjustment_config[
        [
            "adjustment_code",
            "label",
            "multiplier_add",
            "min_bound",
            "max_bound",
            "is_active",
            "updated_at",
        ]
    ].rename(
        columns={
            "adjustment_code": "adjustment_flag",
            "multiplier_add": "multiplier",
        }
    )


def validate_class_config_edit(edited_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the editable class config slice before persisting.
    """
    expected_cols = [
        "task_class",
        "class_name",
        "base_class_weight",
        "base_expected_hours",
        "is_active",
        "updated_at",
    ]
    if edited_df.columns.tolist() != expected_cols:
        raise ValueError(
            "class_config editor columns must remain exactly: "
            "task_class, class_name, base_class_weight, base_expected_hours, "
            "is_active, updated_at"
        )

    validated = edited_df.copy()
    validated["task_class"] = validated["task_class"].astype(str).str.strip()

    if (validated["task_class"] == "").any():
        raise ValueError("task_class cannot be blank.")
    if validated["task_class"].duplicated().any():
        raise ValueError("task_class values must be unique.")

    for col in ["base_class_weight", "base_expected_hours"]:
        validated[col] = pd.to_numeric(validated[col], errors="raise")
        if (validated[col] <= 0).any():
            raise ValueError(f"{col} must be numeric and greater than 0.")

    return validated


def validate_adjustment_config_edit(
    edited_df: pd.DataFrame, adjustment_config: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate the editable adjustment config slice before persisting.
    """
    expected_cols = [
        "adjustment_flag",
        "label",
        "multiplier",
        "min_bound",
        "max_bound",
        "is_active",
        "updated_at",
    ]
    if edited_df.columns.tolist() != expected_cols:
        raise ValueError(
            "adjustment_config editor columns must remain exactly: "
            "adjustment_flag, label, multiplier, min_bound, max_bound, is_active, updated_at"
        )

    validated = edited_df.copy()
    validated["adjustment_flag"] = validated["adjustment_flag"].astype(str).str.strip()

    if (validated["adjustment_flag"] == "").any():
        raise ValueError("adjustment_flag cannot be blank.")
    if validated["adjustment_flag"].duplicated().any():
        raise ValueError("adjustment_flag values must be unique.")

    validated["multiplier"] = pd.to_numeric(validated["multiplier"], errors="raise")
    if (validated["multiplier"] <= 0).any():
        raise ValueError("multiplier must be numeric and greater than 0.")

    bounds = adjustment_config.set_index("adjustment_code")[["min_bound", "max_bound"]]
    missing_flags = set(validated["adjustment_flag"]) - set(bounds.index)
    if missing_flags:
        raise ValueError(
            f"adjustment_flag values not found in adjustment_config.csv: {sorted(missing_flags)}"
        )

    for row in validated.itertuples(index=False):
        min_bound = float(bounds.loc[row.adjustment_flag, "min_bound"])
        max_bound = float(bounds.loc[row.adjustment_flag, "max_bound"])
        if row.multiplier < min_bound or row.multiplier > max_bound:
            raise ValueError(
                f"Adjustment '{row.adjustment_flag}' multiplier {row.multiplier} "
                f"is outside the allowed range [{min_bound}, {max_bound}]."
            )

    return validated


def get_class_config_changes(
    current_df: pd.DataFrame, edited_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Return only changed class config rows for review and save decisions.
    """
    current = current_df[["task_class", "base_class_weight", "base_expected_hours"]].copy()
    edited = edited_df[["task_class", "base_class_weight", "base_expected_hours"]].copy()

    current = current.set_index("task_class").sort_index()
    edited = edited.set_index("task_class").sort_index()

    changed_mask = (current != edited).any(axis=1)
    changed_ids = changed_mask.index[changed_mask]

    if len(changed_ids) == 0:
        return pd.DataFrame(
            columns=[
                "task_class",
                "old_base_class_weight",
                "new_base_class_weight",
                "old_base_expected_hours",
                "new_base_expected_hours",
            ]
        )

    preview = pd.DataFrame(
        {
            "task_class": changed_ids,
            "old_base_class_weight": current.loc[changed_ids, "base_class_weight"].values,
            "new_base_class_weight": edited.loc[changed_ids, "base_class_weight"].values,
            "old_base_expected_hours": current.loc[changed_ids, "base_expected_hours"].values,
            "new_base_expected_hours": edited.loc[changed_ids, "base_expected_hours"].values,
        }
    )
    return preview.reset_index(drop=True)


def get_adjustment_config_changes(
    current_df: pd.DataFrame, edited_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Return only changed adjustment config rows for review and save decisions.
    """
    current = build_adjustment_config_editor_df(current_df)[
        ["adjustment_flag", "multiplier", "min_bound", "max_bound"]
    ].copy()
    edited = edited_df[["adjustment_flag", "multiplier"]].copy()

    current = current.set_index("adjustment_flag").sort_index()
    edited = edited.set_index("adjustment_flag").sort_index()

    changed_mask = current[["multiplier"]] != edited[["multiplier"]]
    changed_ids = changed_mask.index[changed_mask["multiplier"]]

    if len(changed_ids) == 0:
        return pd.DataFrame(
            columns=[
                "adjustment_flag",
                "old_multiplier",
                "new_multiplier",
                "min_bound",
                "max_bound",
            ]
        )

    preview = pd.DataFrame(
        {
            "adjustment_flag": changed_ids,
            "old_multiplier": current.loc[changed_ids, "multiplier"].values,
            "new_multiplier": edited.loc[changed_ids, "multiplier"].values,
            "min_bound": current.loc[changed_ids, "min_bound"].values,
            "max_bound": current.loc[changed_ids, "max_bound"].values,
        }
    )
    return preview.reset_index(drop=True)


def save_class_config(edited_df: pd.DataFrame) -> bool:
    """
    Persist validated class config edits back to the existing CSV.
    """
    validated = validate_class_config_edit(edited_df)
    class_config = load_class_config()
    changes = get_class_config_changes(class_config, validated)
    if changes.empty:
        return False

    updated = class_config.copy()

    updated = updated.merge(validated, on="task_class", how="left", suffixes=("", "_edited"))
    updated["base_class_weight"] = updated["base_class_weight_edited"]
    updated["base_expected_hours"] = updated["base_expected_hours_edited"]
    updated = updated.drop(
        columns=["base_class_weight_edited", "base_expected_hours_edited"]
    )

    if "updated_at" in updated.columns:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated.loc[
            updated["task_class"].isin(changes["task_class"]), "updated_at"
        ] = timestamp

    updated.to_csv(CONFIG_DIR / "class_config.csv", index=False)
    return True


def save_adjustment_config(edited_df: pd.DataFrame) -> bool:
    """
    Persist validated adjustment config edits back to the existing CSV.
    """
    adjustment_config = load_adjustment_config()
    validated = validate_adjustment_config_edit(edited_df, adjustment_config)
    changes = get_adjustment_config_changes(adjustment_config, validated)
    if changes.empty:
        return False

    updated = adjustment_config.copy()

    updated = updated.merge(
        validated.rename(
            columns={
                "adjustment_flag": "adjustment_code",
                "multiplier": "multiplier_add",
            }
        ),
        on="adjustment_code",
        how="left",
        suffixes=("", "_edited"),
    )
    updated["multiplier_add"] = updated["multiplier_add_edited"]
    updated = updated.drop(columns=["multiplier_add_edited"])

    if "updated_at" in updated.columns:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated.loc[
            updated["adjustment_code"].isin(changes["adjustment_flag"]), "updated_at"
        ] = timestamp

    updated.to_csv(CONFIG_DIR / "adjustment_config.csv", index=False)
    return True


def format_task_table(task_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Build a cleaner display table for Streamlit.
    """
    display_cols = [
        "task_id",
        "period",
        "task_class",
        "active_adjustments",
        "base_class_weight",
        "base_expected_hours",
        "adjustment_multiplier",
        "expected_time_hours",
        "actual_time_hours",
        "efficiency_ratio_capped",
        "minor_errors",
        "major_errors",
        "weighted_errors",
        "task_output",
    ]

    df = task_metrics[display_cols].copy()

    rename_map = {
        "task_id": "Task ID",
        "period": "Period",
        "task_class": "Class",
        "active_adjustments": "Adjustments",
        "base_class_weight": "Base Weight",
        "base_expected_hours": "Base Expected Hours",
        "adjustment_multiplier": "Adjustment Multiplier",
        "expected_time_hours": "Expected Hours",
        "actual_time_hours": "Actual Hours",
        "efficiency_ratio_capped": "Efficiency Ratio",
        "minor_errors": "Minor Errors",
        "major_errors": "Major Errors",
        "weighted_errors": "Weighted Errors",
        "task_output": "Task Output",
    }

    df = df.rename(columns=rename_map)

    numeric_cols = [
        "Base Weight",
        "Base Expected Hours",
        "Adjustment Multiplier",
        "Expected Hours",
        "Actual Hours",
        "Efficiency Ratio",
        "Weighted Errors",
        "Task Output",
    ]
    df[numeric_cols] = df[numeric_cols].round(4)

    return df


def format_attribution_records(records: list[dict], rename_map: dict | None = None) -> pd.DataFrame:
    """
    Build a display dataframe for attribution records.
    """
    df = pd.DataFrame(records).copy()
    if df.empty:
        return df
    if rename_map:
        df = df.rename(columns=rename_map)

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].round(4)
    return df


def build_cross_intern_summary(results_by_intern: dict[str, ScoreResults]) -> pd.DataFrame:
    """
    Build a display-only cross-intern summary from existing scoring outputs.
    """
    rows = []
    for intern_id in sorted(results_by_intern.keys()):
        summary = results_by_intern[intern_id].summary
        rows.append(
            {
                "intern_id": str(intern_id),
                "final_score": summary["final_score"],
                "performance_index": summary["performance_index"],
                "output_score": summary["output_score"],
                "efficiency_score": summary["efficiency_score"],
                "accuracy_score": summary["accuracy_score"],
                "contribution_modifier": summary["contribution_modifier"],
                "total_weighted_errors": summary["total_weighted_errors"],
            }
        )

    comparison_df = pd.DataFrame(rows)
    return comparison_df


def build_ranked_leaderboard(comparison_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Rank interns descending by a selected summary metric.
    """
    ranked = comparison_df.sort_values(
        [metric, "intern_id"], ascending=[False, True], kind="mergesort"
    ).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def build_distribution_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-intern min/max/mean for selected metrics.
    """
    metric_cols = [
        "efficiency_score",
        "accuracy_score",
        "contribution_modifier",
        "performance_index",
    ]
    rows = []
    for metric in metric_cols:
        rows.append(
            {
                "metric": metric,
                "min": comparison_df[metric].min(),
                "max": comparison_df[metric].max(),
                "mean": comparison_df[metric].mean(),
            }
        )
    return pd.DataFrame(rows)


def identify_widest_metric_variance(comparison_df: pd.DataFrame) -> dict[str, str | float]:
    """
    Identify the metric with the widest cross-intern spread (max - min).
    """
    metric_cols = [
        "final_score",
        "performance_index",
        "output_score",
        "efficiency_score",
        "accuracy_score",
        "contribution_modifier",
    ]

    widest = {
        "metric": "",
        "spread": -1.0,
        "max_intern_id": "",
        "max_value": 0.0,
        "min_intern_id": "",
        "min_value": 0.0,
    }

    for metric in metric_cols:
        max_idx = comparison_df[metric].idxmax()
        min_idx = comparison_df[metric].idxmin()
        max_row = comparison_df.loc[max_idx]
        min_row = comparison_df.loc[min_idx]
        spread = float(max_row[metric] - min_row[metric])

        if spread > widest["spread"]:
            widest = {
                "metric": metric,
                "spread": spread,
                "max_intern_id": str(max_row["intern_id"]),
                "max_value": float(max_row[metric]),
                "min_intern_id": str(min_row["intern_id"]),
                "min_value": float(min_row[metric]),
            }

    return widest


def identify_cross_intern_outliers(comparison_df: pd.DataFrame) -> dict[str, dict]:
    """
    Identify requested outlier rows from existing summary fields only.
    """
    lowest_efficiency = comparison_df.loc[comparison_df["efficiency_score"].idxmin()].to_dict()
    highest_error_burden = comparison_df.loc[
        comparison_df["total_weighted_errors"].idxmax()
    ].to_dict()
    weakest_contribution = comparison_df.loc[
        comparison_df["contribution_modifier"].idxmin()
    ].to_dict()
    widest_metric_variance = identify_widest_metric_variance(comparison_df)

    top_performer = comparison_df.loc[comparison_df["final_score"].idxmax()].to_dict()
    lowest_performer = comparison_df.loc[comparison_df["final_score"].idxmin()].to_dict()

    return {
        "top_performer": top_performer,
        "lowest_performer": lowest_performer,
        "widest_metric_variance": widest_metric_variance,
        "lowest_efficiency": lowest_efficiency,
        "highest_error_burden": highest_error_burden,
        "weakest_contribution": weakest_contribution,
    }


def render_cross_intern_insights(results_by_intern: dict[str, ScoreResults]) -> None:
    st.header("Cross-Intern Insights")
    st.caption("Read-only comparative layer derived from run_scoring() outputs.")

    comparison_df = build_cross_intern_summary(results_by_intern)
    if comparison_df.empty:
        st.info("No intern results available for comparison.")
        return

    rank_metric = st.selectbox(
        "Leaderboard ranking metric",
        options=[
            "final_score",
            "performance_index",
            "output_score",
            "efficiency_score",
            "accuracy_score",
            "contribution_modifier",
        ],
        index=0,
    )

    st.subheader("Ranked Leaderboard")
    leaderboard = build_ranked_leaderboard(comparison_df, rank_metric)
    st.dataframe(
        leaderboard[
            [
                "rank",
                "intern_id",
                "final_score",
                "performance_index",
                "output_score",
                "efficiency_score",
                "accuracy_score",
                "contribution_modifier",
            ]
        ].round(4),
        hide_index=True,
        use_container_width=True,
    )

    insights = identify_cross_intern_outliers(comparison_df)
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Top Performer (final_score)",
        str(insights["top_performer"]["intern_id"]),
        f"{insights['top_performer']['final_score']:.4f}",
    )
    c2.metric(
        "Lowest Performer (final_score)",
        str(insights["lowest_performer"]["intern_id"]),
        f"{insights['lowest_performer']['final_score']:.4f}",
    )
    c3.metric(
        "Widest Variance Across Metrics",
        str(insights["widest_metric_variance"]["metric"]),
        f"{insights['widest_metric_variance']['spread']:.4f}",
    )
    st.caption(
        f"Max: {insights['widest_metric_variance']['max_intern_id']} "
        f"({insights['widest_metric_variance']['max_value']:.4f}) | "
        f"Min: {insights['widest_metric_variance']['min_intern_id']} "
        f"({insights['widest_metric_variance']['min_value']:.4f})"
    )

    st.subheader("Metric Comparison Table")
    st.dataframe(
        comparison_df[
            [
                "intern_id",
                "final_score",
                "performance_index",
                "output_score",
                "efficiency_score",
                "accuracy_score",
                "contribution_modifier",
                "total_weighted_errors",
            ]
        ].round(4),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Distribution Summary")
    distributions = build_distribution_summary(comparison_df).round(4)
    st.dataframe(distributions, hide_index=True, use_container_width=True)

    st.subheader("Outlier Highlights")
    o1, o2, o3 = st.columns(3)
    o1.metric(
        "Lowest efficiency_score",
        str(insights["lowest_efficiency"]["intern_id"]),
        f"{insights['lowest_efficiency']['efficiency_score']:.4f}",
    )
    o2.metric(
        "Highest total_weighted_errors",
        str(insights["highest_error_burden"]["intern_id"]),
        f"{insights['highest_error_burden']['total_weighted_errors']:.4f}",
    )
    o3.metric(
        "Weakest contribution_modifier",
        str(insights["weakest_contribution"]["intern_id"]),
        f"{insights['weakest_contribution']['contribution_modifier']:.4f}",
    )


def render_diagnostic_insights(results_by_intern: dict[str, ScoreResults], intern_id: str) -> None:
    st.header("Diagnostic Insights")
    st.caption("Deterministic interpretation layer derived from summary, attribution, and peer context.")

    insights = build_diagnostic_insights(results_by_intern, intern_id)
    intern_summary = insights["intern_summary"]
    positioning = insights["cross_intern_positioning"]
    attribution_explanations = insights["attribution_explanations"]

    st.subheader("Intern Diagnostic Summary")
    st.write(f"- **Primary strength driver:** {intern_summary['primary_strength_driver']}")
    st.write(f"- **Primary weakness driver:** {intern_summary['primary_weakness_driver']}")
    st.write(f"- **Dominant final score driver:** {intern_summary['dominant_final_score_driver']}")
    st.write(
        f"- **Performance index interpretation:** "
        f"{intern_summary['performance_index_interpretation']}"
    )

    st.divider()

    st.subheader("Cross-Intern Positioning")
    p1, p2 = st.columns(2)
    p1.metric(
        "Final Score Rank",
        f"{positioning['final_score_rank']} / {positioning['final_score_total']}",
    )
    p2.metric(
        "Performance Index Rank",
        f"{positioning['performance_index_rank']} / {positioning['performance_index_total']}",
    )
    st.write(f"- {positioning['final_score_positioning']}")
    st.write(f"- {positioning['performance_index_positioning']}")
    st.write(f"- {positioning['strongest_peer_advantage']}")
    st.write(f"- {positioning['largest_peer_gap']}")
    for highlight in positioning["peer_comparison_highlights"]:
        st.write(f"- {highlight}")

    st.divider()

    st.subheader("Attribution-Based Highlights")
    st.write(f"- **Output:** {attribution_explanations['output_explanation']}")
    st.write(f"- **Efficiency:** {attribution_explanations['efficiency_explanation']}")
    st.write(f"- **Accuracy:** {attribution_explanations['accuracy_explanation']}")
    st.write(f"- **Contribution:** {attribution_explanations['contribution_explanation']}")


def render_scenario_simulation() -> None:
    st.subheader("Scenario Simulation")
    st.caption("Sandboxed what-if run. Edits are applied in memory only.")

    class_config = load_class_config()
    adjustment_config = load_adjustment_config()
    saved_scenarios = list_scenarios()

    st.write("Saved Scenarios")
    if saved_scenarios:
        scenario_options = [row["scenario_id"] for row in saved_scenarios]
        selected_scenario_id = st.selectbox(
            "Saved scenario",
            options=scenario_options,
            format_func=lambda sid: next(
                row["scenario_name"] for row in saved_scenarios if row["scenario_id"] == sid
            ),
            key="saved_scenario_selector",
        )
        s1, s2, s3 = st.columns(3)
        if s1.button("Load Scenario", key="load_saved_scenario"):
            try:
                selected_record = load_scenario(selected_scenario_id)
                st.session_state["loaded_scenario_id"] = selected_record["scenario_id"]
                st.session_state["loaded_scenario_name"] = selected_record["scenario_name"]
                st.session_state["loaded_scenario_overrides"] = selected_record["overrides"]
                st.rerun()
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
        if s2.button("Run Saved Scenario", key="run_saved_scenario"):
            try:
                saved_run = run_saved_scenario(selected_scenario_id)
                render_simulation_result(
                    saved_run["simulation_result"], saved_run["deltas"]
                )
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))
        if s3.button("Delete Scenario", key="delete_saved_scenario"):
            if delete_scenario(selected_scenario_id):
                st.success(f"Deleted scenario {selected_scenario_id}.")
                st.rerun()
            st.warning(f"Scenario {selected_scenario_id} was not found.")

        st.dataframe(
            pd.DataFrame(saved_scenarios),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No saved scenarios yet.")

    loaded_overrides = st.session_state.get("loaded_scenario_overrides")
    loaded_name = st.session_state.get("loaded_scenario_name", "Manager what-if scenario")
    class_editor_source = class_config[
        ["task_class", "class_name", "base_expected_hours", "base_class_weight"]
    ].copy()
    adjustment_editor_source = adjustment_config[
        ["adjustment_code", "label", "multiplier_add", "min_bound", "max_bound"]
    ].copy()
    if loaded_overrides:
        class_expected = loaded_overrides["class_expected_hours_overrides"]
        class_weights = loaded_overrides["class_weight_overrides"]
        adjustment_multipliers = loaded_overrides["adjustment_multiplier_overrides"]
        for task_class, value in class_expected.items():
            class_editor_source.loc[
                class_editor_source["task_class"] == task_class, "base_expected_hours"
            ] = value
        for task_class, value in class_weights.items():
            class_editor_source.loc[
                class_editor_source["task_class"] == task_class, "base_class_weight"
            ] = value
        for adjustment_code, value in adjustment_multipliers.items():
            adjustment_editor_source.loc[
                adjustment_editor_source["adjustment_code"] == adjustment_code,
                "multiplier_add",
            ] = value

    editor_key_suffix = st.session_state.get("loaded_scenario_id", "draft")

    with st.form("scenario_simulation_form"):
        scenario_name = st.text_input("Scenario name", value=loaded_name)

        st.write("Class Expected Hours and Weights")
        class_editor = st.data_editor(
            class_editor_source,
            hide_index=True,
            use_container_width=True,
            disabled=["task_class", "class_name"],
            key=f"scenario_class_editor_{editor_key_suffix}",
        )

        st.write("Adjustment Multipliers")
        adjustment_editor = st.data_editor(
            adjustment_editor_source,
            hide_index=True,
            use_container_width=True,
            disabled=["adjustment_code", "label", "min_bound", "max_bound"],
            key=f"scenario_adjustment_editor_{editor_key_suffix}",
        )

        save_after_run = st.checkbox("Save scenario after run", value=False)
        overwrite_existing = st.checkbox("Overwrite existing scenario", value=False)
        submitted = st.form_submit_button("Run Simulation")

    if not submitted:
        return

    base_class = class_config.set_index("task_class")
    edited_class = class_editor.set_index("task_class")
    expected_hours_overrides = {}
    class_weight_overrides = {}
    for task_class in edited_class.index:
        if edited_class.loc[task_class, "base_expected_hours"] != base_class.loc[task_class, "base_expected_hours"]:
            expected_hours_overrides[str(task_class)] = edited_class.loc[
                task_class, "base_expected_hours"
            ]
        if edited_class.loc[task_class, "base_class_weight"] != base_class.loc[task_class, "base_class_weight"]:
            class_weight_overrides[str(task_class)] = edited_class.loc[
                task_class, "base_class_weight"
            ]

    base_adjustment = adjustment_config.set_index("adjustment_code")
    edited_adjustment = adjustment_editor.set_index("adjustment_code")
    adjustment_overrides = {}
    for adjustment_code in edited_adjustment.index:
        if edited_adjustment.loc[adjustment_code, "multiplier_add"] != base_adjustment.loc[adjustment_code, "multiplier_add"]:
            adjustment_overrides[str(adjustment_code)] = edited_adjustment.loc[
                adjustment_code, "multiplier_add"
            ]

    try:
        if save_after_run:
            saved_record = save_scenario(
                scenario_name=scenario_name,
                class_expected_hours_overrides=expected_hours_overrides,
                class_weight_overrides=class_weight_overrides,
                adjustment_multiplier_overrides=adjustment_overrides,
                overwrite=overwrite_existing,
            )
            st.success(f"Saved scenario {saved_record['scenario_id']}.")

        simulation_result = run_simulation(
            scenario_name=scenario_name,
            class_expected_hours_overrides=expected_hours_overrides,
            class_weight_overrides=class_weight_overrides,
            adjustment_multiplier_overrides=adjustment_overrides,
        )
        deltas = build_simulation_deltas(
            simulation_result["baseline"], simulation_result["simulated"]
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    render_simulation_result(simulation_result, deltas)


def render_simulation_result(simulation_result: dict, deltas: dict) -> None:
    baseline_scores = simulation_result["baseline"]["scores"]
    simulated_scores = simulation_result["simulated"]["scores"]
    comparison_rows = []
    for intern_id in sorted(baseline_scores.keys()):
        baseline_summary = baseline_scores[intern_id].summary
        simulated_summary = simulated_scores[intern_id].summary
        comparison_rows.append(
            {
                "Intern ID": str(intern_id),
                "Baseline Final Score": float(baseline_summary["final_score"]),
                "Simulated Final Score": float(simulated_summary["final_score"]),
                "Baseline Category": baseline_summary["performance_category"],
                "Simulated Category": simulated_summary["performance_category"],
            }
        )

    st.write(f"Overrides applied: {simulation_result['scenario_metadata']['override_count']}")
    st.dataframe(
        pd.DataFrame(comparison_rows).round(4),
        hide_index=True,
        use_container_width=True,
    )

    baseline_patterns = simulation_result["baseline"]["system_patterns"]["pattern_summary"]
    simulated_patterns = simulation_result["simulated"]["system_patterns"]["pattern_summary"]
    baseline_actions = simulation_result["baseline"]["manager_actions"]["action_summary"]
    simulated_actions = simulation_result["simulated"]["manager_actions"]["action_summary"]

    a1, a2, p1, p2 = st.columns(4)
    a1.metric(
        "Baseline Actions",
        int(baseline_actions["total_intern_actions"] + baseline_actions["total_team_actions"]),
    )
    a2.metric(
        "Simulated Actions",
        int(simulated_actions["total_intern_actions"] + simulated_actions["total_team_actions"]),
    )
    p1.metric("Baseline Patterns", int(baseline_patterns["total_patterns"]))
    p2.metric("Simulated Patterns", int(simulated_patterns["total_patterns"]))

    st.write("Delta Summary")
    changed_scores = [
        row
        for row in deltas["metric_deltas"]
        if row["metric_name"] == "final_score" and row["direction"] != "no_change"
    ]
    category_transitions = [
        row for row in deltas["category_changes"] if row["changed"]
    ]
    action_transitions = [
        row
        for row in deltas["action_changes"]
        if row["change_type"] in {"added", "removed", "priority_changed"}
    ]
    pattern_scope_shifts = [
        row
        for row in deltas["pattern_changes"]
        if row["change_type"] in {"scope_changed", "introduced", "resolved"}
    ]

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Changed Scores", len(changed_scores))
    d2.metric("Category Transitions", len(category_transitions))
    d3.metric("Action Changes", len(action_transitions))
    d4.metric("Pattern Scope Shifts", len(pattern_scope_shifts))

    if changed_scores:
        st.dataframe(
            pd.DataFrame(changed_scores).round(4),
            hide_index=True,
            use_container_width=True,
        )
    if category_transitions:
        st.dataframe(
            pd.DataFrame(category_transitions),
            hide_index=True,
            use_container_width=True,
        )
    if action_transitions:
        st.dataframe(
            pd.DataFrame(action_transitions),
            hide_index=True,
            use_container_width=True,
        )
    if pattern_scope_shifts:
        st.dataframe(
            pd.DataFrame(pattern_scope_shifts).round(4),
            hide_index=True,
            use_container_width=True,
        )


def render_manager_view(results_by_intern: dict[str, ScoreResults], default_intern_id: str) -> None:
    st.header("Manager View")
    st.caption("Decision-first manager summary built from existing deterministic analytics outputs.")

    comparison_df = build_cross_intern_summary(results_by_intern)
    if comparison_df.empty:
        st.info("No intern results available.")
        return

    patterns_payload = build_cross_intern_patterns(results_by_intern)
    actions_payload = build_manager_actions(results_by_intern)

    # Executive Summary
    top_idx = comparison_df["final_score"].idxmax()
    low_idx = comparison_df["final_score"].idxmin()
    top_intern = comparison_df.loc[top_idx]
    low_intern = comparison_df.loc[low_idx]
    summary = patterns_payload["pattern_summary"]
    action_summary = actions_payload["action_summary"]

    st.subheader("Executive Summary")
    e1, e2, e3, e4, e5, e6 = st.columns(6)
    e1.metric("Total Interns", int(len(comparison_df)))
    e2.metric("Top Performer", str(top_intern["intern_id"]), f"{float(top_intern['final_score']):.4f}")
    e3.metric("Lowest Performer", str(low_intern["intern_id"]), f"{float(low_intern['final_score']):.4f}")
    e4.metric("High-Priority Actions", int(action_summary["high_priority_count"]))
    e5.metric("Systemic Patterns", int(summary["systemic_count"]))
    e6.metric("Emerging Patterns", int(summary["emerging_count"]))

    st.divider()

    render_scenario_simulation()

    st.divider()

    # Priority Action Queue
    st.subheader("Priority Action Queue")
    all_actions = actions_payload["intern_actions"] + actions_payload["team_actions"]
    priority_filter = st.multiselect(
        "Filter action priorities",
        options=["high", "moderate", "low"],
        default=["high", "moderate", "low"],
        key="manager_action_priority_filter",
    )
    scope_filter = st.multiselect(
        "Filter action scopes",
        options=["intern", "team", "system"],
        default=["intern", "team", "system"],
        key="manager_action_scope_filter",
    )

    priority_order = {"high": 0, "moderate": 1, "low": 2}
    filtered_actions = [
        action
        for action in all_actions
        if action["priority_level"] in priority_filter and action["target_scope"] in scope_filter
    ]
    filtered_actions = sorted(
        filtered_actions,
        key=lambda action: (
            priority_order.get(action["priority_level"], 99),
            action["target_scope"],
            action["action_type"],
            action["target_id"],
            action["action_key"],
        ),
    )

    if not filtered_actions:
        st.info("No actions match the selected filters.")
    else:
        action_rows = []
        for action in filtered_actions:
            action_rows.append(
                {
                    "Priority": action["priority_level"],
                    "Scope": action["target_scope"],
                    "Type": action["action_type"],
                    "Target": action["target_id"],
                    "Message": action["message"],
                    "Rationale": action["rationale"],
                    "Evidence Sources": ", ".join(action["evidence_sources"]),
                }
            )
        st.dataframe(pd.DataFrame(action_rows), hide_index=True, use_container_width=True)

    st.divider()

    # Team/System Pattern Highlights
    st.subheader("Team/System Pattern Highlights")
    scope_class_filter = st.multiselect(
        "Filter pattern scope classification",
        options=["systemic", "emerging", "isolated"],
        default=["systemic", "emerging"],
        key="manager_pattern_scope_filter",
    )
    pattern_rows = []
    for pattern in patterns_payload["system_patterns"]:
        if pattern["scope_classification"] not in scope_class_filter:
            continue
        pattern_rows.append(
            {
                "Pattern Type": pattern["pattern_type"],
                "Message": pattern["message"],
                "Frequency": round(float(pattern["frequency"]), 4),
                "Intern Coverage": f"{pattern['intern_count']}/{pattern['total_interns']}",
                "Scope": pattern["scope_classification"],
                "Severity": pattern["severity"],
            }
        )

    if pattern_rows:
        st.dataframe(pd.DataFrame(pattern_rows), hide_index=True, use_container_width=True)
    else:
        st.info("No patterns match the selected scope filter.")

    st.divider()

    # Intern Risk / Strength Snapshot
    st.subheader("Intern Risk / Strength Snapshot")
    top_action_by_intern: dict[str, dict] = {}
    for action in actions_payload["intern_actions"]:
        iid = str(action["target_id"])
        if iid not in top_action_by_intern:
            top_action_by_intern[iid] = action

    snapshot_rows = []
    for intern_id in sorted(results_by_intern.keys()):
        intern_summary = results_by_intern[intern_id].summary
        insights = build_diagnostic_insights(results_by_intern, intern_id)
        top_action = top_action_by_intern.get(str(intern_id))
        snapshot_rows.append(
            {
                "Intern ID": str(intern_id),
                "Category": intern_summary["performance_category"],
                "Final Score": round(float(intern_summary["final_score"]), 4),
                "Primary Strength": insights["intern_summary"]["primary_strength_driver"],
                "Primary Weakness": insights["intern_summary"]["primary_weakness_driver"],
                "Top Action Priority": top_action["priority_level"] if top_action else "none",
                "Top Action Type": top_action["action_type"] if top_action else "none",
                "Action Status": "Action queued" if top_action else "No active action",
            }
        )
    st.dataframe(pd.DataFrame(snapshot_rows), hide_index=True, use_container_width=True)

    st.divider()

    # Drilldown Selector
    st.subheader("Intern Drilldown")
    drilldown_intern_id = st.selectbox(
        "Select intern for focused manager summary",
        options=sorted(results_by_intern.keys()),
        index=sorted(results_by_intern.keys()).index(default_intern_id),
        key="manager_drilldown_intern",
    )
    selected = results_by_intern[drilldown_intern_id]
    selected_summary = selected.summary
    selected_insights = build_diagnostic_insights(results_by_intern, drilldown_intern_id)

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Final Score", f"{float(selected_summary['final_score']):.4f}")
    d2.metric("Performance Index", f"{float(selected_summary['performance_index']):.4f}")
    d3.metric("Category", str(selected_summary["performance_category"]))
    d4.metric("Total Tasks", int(selected_summary["total_tasks"]))

    st.write(f"- **Primary strength:** {selected_insights['intern_summary']['primary_strength_driver']}")
    st.write(f"- **Primary weakness:** {selected_insights['intern_summary']['primary_weakness_driver']}")
    st.write(
        f"- **Dominant final-score driver:** "
        f"{selected_insights['intern_summary']['dominant_final_score_driver']}"
    )

    selected_actions = [
        action
        for action in actions_payload["intern_actions"]
        if str(action["target_id"]) == str(drilldown_intern_id)
    ]
    st.write("**Top Intern Actions**")
    if selected_actions:
        for action in selected_actions[:3]:
            st.write(
                f"- [{action['priority_level']}] {action['action_type']}: "
                f"{action['message']} ({action['rationale']})"
            )
    else:
        st.write("- No intern-specific manager actions currently queued.")

    attr = selected_insights["attribution_explanations"]
    st.write("**Key Attribution Highlights**")
    st.write(f"- Output: {attr['output_explanation']}")
    st.write(f"- Efficiency: {attr['efficiency_explanation']}")
    st.write(f"- Accuracy: {attr['accuracy_explanation']}")
    st.write(f"- Contribution: {attr['contribution_explanation']}")


def render_overview(summary: dict) -> None:
    st.title("First-Year Tax Intern Performance Dashboard")
    st.caption("Busy season MVP demo with batch scoring and selected-intern view.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Intern Overview")
        st.write(f"**Intern ID:** {summary['intern_id']}")
        st.write(f"**Total Tasks:** {summary['total_tasks']}")
        st.write(f"**Performance Category:** {summary['performance_category']}")

    with col2:
        st.subheader("Scoring Formula")
        st.code(
            "Final Score = Output × Efficiency × Accuracy × Contribution",
            language="text",
        )

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Final Score", f"{summary['final_score']:.4f}")
    m2.metric("Output", f"{summary['output_score']:.4f}")
    m3.metric("Efficiency", f"{summary['efficiency_score']:.4f}")
    m4.metric("Accuracy", f"{summary['accuracy_score']:.4f}")
    m5.metric("Contribution", f"{summary['contribution_modifier']:.4f}")

    st.divider()

    st.subheader("Interpretation")
    st.write(
        "This score is driven by weighted output volume, time efficiency, review accuracy, "
        "and a capped contribution modifier from structured flags."
    )

    st.subheader("Score Breakdown")
    st.write(
        "Final Score is calculated as Output Score × Efficiency Score × Accuracy Score × "
        "Contribution Modifier."
    )

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Output Score", f"{summary['output_score']:.4f}")
    b2.metric("Efficiency Score", f"{summary['efficiency_score']:.4f}")
    b3.metric("Accuracy Score", f"{summary['accuracy_score']:.4f}")
    b4.metric("Contribution Modifier", f"{summary['contribution_modifier']:.4f}")
    b5.metric("Final Score", f"{summary['final_score']:.4f}")

    st.code(
        f"{summary['output_score']:.4f} × {summary['efficiency_score']:.4f} × "
        f"{summary['accuracy_score']:.4f} × {summary['contribution_modifier']:.4f} = "
        f"{summary['final_score']:.4f}",
        language="text",
    )


def render_task_breakdown(task_metrics: pd.DataFrame) -> None:
    st.header("Task Breakdown")

    class_filter = st.multiselect(
        "Filter by class",
        options=sorted(task_metrics["task_class"].unique().tolist()),
        default=sorted(task_metrics["task_class"].unique().tolist()),
    )

    period_filter = st.multiselect(
        "Filter by period",
        options=sorted(task_metrics["period"].unique().tolist()),
        default=sorted(task_metrics["period"].unique().tolist()),
    )

    error_filter = st.selectbox(
        "Filter by error status",
        options=["All", "Has Errors", "No Errors"],
        index=0,
    )

    filtered = task_metrics[
        task_metrics["task_class"].isin(class_filter)
        & task_metrics["period"].isin(period_filter)
    ].copy()

    if error_filter == "Has Errors":
        filtered = filtered[filtered["weighted_errors"] > 0]
    elif error_filter == "No Errors":
        filtered = filtered[filtered["weighted_errors"] == 0]

    st.dataframe(format_task_table(filtered), use_container_width=True)

    st.subheader("Expected vs Actual Hours")
    chart_df = filtered[["task_id", "expected_time_hours", "actual_time_hours"]].copy()
    chart_df = chart_df.set_index("task_id")
    st.bar_chart(chart_df)


def render_flags_diagnostics(summary: dict, task_metrics: pd.DataFrame, attribution: dict) -> None:
    st.header("Flags & Diagnostics")

    c1, c2, c3 = st.columns(3)
    c1.metric("Positive Flags", summary["positive_flags"])
    c2.metric("Negative Flags", summary["negative_flags"])
    c3.metric("Performance Index", f"{summary['performance_index']:.4f}")

    diagnostics = summary["diagnostics"]

    st.subheader("Diagnostic Notes")
    st.write(f"- {diagnostics['efficiency_diagnostic']}")
    st.write(f"- {diagnostics['accuracy_diagnostic']}")
    st.write(f"- {diagnostics['contribution_diagnostic']}")
    st.write(f"- {diagnostics['overall_diagnostic']}")

    st.divider()

    st.subheader("Efficiency Explanation")
    e1, e2, e3 = st.columns(3)
    e1.metric("Expected Hours", f"{summary['total_expected_time']:.4f}")
    e2.metric("Actual Hours", f"{summary['total_actual_time']:.4f}")
    e3.metric("Efficiency Score", f"{summary['efficiency_score']:.4f}")
    st.write(
        "Efficiency Score is calculated as total expected hours divided by total actual "
        "hours, then capped between 0.5 and 1.25."
    )

    st.subheader("Accuracy Explanation")
    a1, a2, a3 = st.columns(3)
    a1.metric("Total Tasks", f"{summary['total_tasks']}")
    a2.metric("Total Weighted Errors", f"{summary['total_weighted_errors']:.4f}")
    a3.metric("Accuracy Score", f"{summary['accuracy_score']:.4f}")
    st.write(
        "Accuracy Score is calculated as 1 minus total weighted errors divided by total tasks."
    )

    st.subheader("Contribution Explanation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Positive Flags", summary["positive_flags"])
    c2.metric("Negative Flags", summary["negative_flags"])
    c3.metric("Contribution Modifier", f"{summary['contribution_modifier']:.4f}")
    st.write(
        "Contribution Modifier is calculated from positive and negative flags, then capped "
        "between 0.9 and 1.1."
    )

    st.subheader("Score Assembly")
    st.write(
        "Final Score is calculated as Output Score × Efficiency Score × Accuracy Score × "
        "Contribution Modifier."
    )
    st.write(f"Output Score: {summary['output_score']:.4f}")
    st.write(f"Efficiency Score: {summary['efficiency_score']:.4f}")
    st.write(f"Accuracy Score: {summary['accuracy_score']:.4f}")
    st.write(f"Contribution Modifier: {summary['contribution_modifier']:.4f}")
    st.write(f"Final Score: {summary['final_score']:.4f}")

    st.divider()

    st.header("Diagnostic Attribution")
    overall = attribution["overall_attribution"]
    st.subheader("Overall Attribution Summary")
    for line in overall["summary_lines"]:
        st.write(f"- {line}")

    category_driver = overall["category_driver"]
    st.write(
        f"Current category driver: **{category_driver['component']}** = "
        f"{category_driver['value']:.4f} for category **{category_driver['category']}**."
    )

    output_attr = attribution["output_attribution"]
    with st.expander("Output Attribution", expanded=True):
        reconciliation = output_attr["reconciliation"]
        o1, o2, o3 = st.columns(3)
        o1.metric(
            "Base Output",
            f"{reconciliation['base_output_without_adjustments']:.4f}",
        )
        o2.metric(
            "Adjustment Output Effect",
            f"{reconciliation['adjustment_output_effect']:.4f}",
        )
        o3.metric("Total Output", f"{reconciliation['total_output']:.4f}")

        st.write("Output contribution by task class")
        st.dataframe(
            format_attribution_records(
                output_attr["by_class"],
                {
                    "task_class": "Class",
                    "class_name": "Class Name",
                    "task_count": "Task Count",
                    "output_contribution": "Output Contribution",
                    "output_share": "Output Share",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.write("Output contribution by task")
        st.dataframe(
            format_attribution_records(
                output_attr["by_task"],
                {
                    "task_id": "Task ID",
                    "task_class": "Class",
                    "class_name": "Class Name",
                    "base_class_weight": "Base Weight",
                    "adjustment_multiplier": "Adjustment Multiplier",
                    "task_output": "Task Output",
                    "output_share": "Output Share",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.write("Output effect by adjustment flag")
        st.dataframe(
            format_attribution_records(
                output_attr["by_adjustment"],
                {
                    "adjustment_code": "Adjustment Code",
                    "label": "Label",
                    "task_count": "Task Count",
                    "multiplier_add": "Multiplier Add",
                    "output_effect": "Output Effect",
                    "output_effect_share": "Output Effect Share",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

    efficiency_attr = attribution["efficiency_attribution"]
    with st.expander("Efficiency Attribution", expanded=False):
        reconciliation = efficiency_attr["reconciliation"]
        e1, e2, e3 = st.columns(3)
        e1.metric("Expected Hours", f"{reconciliation['total_expected_time']:.4f}")
        e2.metric("Actual Hours", f"{reconciliation['total_actual_time']:.4f}")
        e3.metric("Total Time Delta", f"{reconciliation['total_time_delta_hours']:.4f}")

        st.write("Largest overruns")
        st.dataframe(
            format_attribution_records(
                efficiency_attr["largest_overruns"],
                {
                    "task_id": "Task ID",
                    "task_class": "Class",
                    "expected_time_hours": "Expected Hours",
                    "actual_time_hours": "Actual Hours",
                    "time_delta_hours": "Actual - Expected",
                    "overrun_hours": "Overrun Hours",
                    "underrun_hours": "Underrun Hours",
                    "inefficiency_share": "Inefficiency Share",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.write("Largest underruns")
        st.dataframe(
            format_attribution_records(
                efficiency_attr["largest_underruns"],
                {
                    "task_id": "Task ID",
                    "task_class": "Class",
                    "expected_time_hours": "Expected Hours",
                    "actual_time_hours": "Actual Hours",
                    "time_delta_hours": "Actual - Expected",
                    "overrun_hours": "Overrun Hours",
                    "underrun_hours": "Underrun Hours",
                    "inefficiency_share": "Inefficiency Share",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.write("Efficiency by task")
        st.dataframe(
            format_attribution_records(
                efficiency_attr["by_task"],
                {
                    "task_id": "Task ID",
                    "task_class": "Class",
                    "expected_time_hours": "Expected Hours",
                    "actual_time_hours": "Actual Hours",
                    "time_delta_hours": "Actual - Expected",
                    "overrun_hours": "Overrun Hours",
                    "underrun_hours": "Underrun Hours",
                    "inefficiency_share": "Inefficiency Share",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

    accuracy_attr = attribution["accuracy_attribution"]
    with st.expander("Accuracy Attribution", expanded=False):
        reconciliation = accuracy_attr["reconciliation"]
        a1, a2, a3 = st.columns(3)
        a1.metric("Weighted Errors", f"{reconciliation['total_weighted_errors']:.4f}")
        a2.metric("Total Tasks", f"{reconciliation['total_tasks']}")
        a3.metric("Accuracy Loss", f"{reconciliation['accuracy_loss']:.4f}")

        st.write("Tasks driving accuracy loss")
        st.dataframe(
            format_attribution_records(
                accuracy_attr["top_error_drivers"],
                {
                    "task_id": "Task ID",
                    "task_class": "Class",
                    "minor_errors": "Minor Errors",
                    "major_errors": "Major Errors",
                    "weighted_errors": "Weighted Errors",
                    "minor_error_impact": "Minor Error Impact",
                    "major_error_impact": "Major Error Impact",
                    "total_error_count": "Total Error Count",
                    "accuracy_loss_share": "Accuracy Loss Share",
                    "accuracy_score_impact": "Accuracy Score Impact",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.write("Error impact by severity")
        st.dataframe(
            format_attribution_records(
                accuracy_attr["by_severity"],
                {
                    "severity": "Severity",
                    "error_count": "Error Count",
                    "weighted_error_impact": "Weighted Error Impact",
                    "accuracy_score_impact": "Accuracy Score Impact",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

    contribution_attr = attribution["contribution_attribution"]
    with st.expander("Contribution Attribution", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Raw Modifier Before Cap",
            f"{contribution_attr['raw_modifier_before_cap']:.4f}",
        )
        c2.metric(
            "Final Modifier After Cap",
            f"{contribution_attr['final_modifier_after_cap']:.4f}",
        )
        c3.metric("Cap Applied", str(contribution_attr["cap_applied"]))

        st.write(
            f"Raw positive effect: {contribution_attr['raw_positive_effect']:.4f}; "
            f"raw negative effect: {contribution_attr['raw_negative_effect']:.4f}."
        )

        st.write("Positive flag effect by type")
        st.dataframe(
            format_attribution_records(
                contribution_attr["positive_by_type"],
                {
                    "flag_type": "Flag Type",
                    "flag_count": "Flag Count",
                    "modifier_effect": "Modifier Effect",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.write("Negative flag effect by type")
        st.dataframe(
            format_attribution_records(
                contribution_attr["negative_by_type"],
                {
                    "flag_type": "Flag Type",
                    "flag_count": "Flag Count",
                    "modifier_effect": "Modifier Effect",
                },
            ),
            hide_index=True,
            use_container_width=True,
        )


def render_admin_controls() -> None:
    st.header("Admin Controls")
    st.caption(
        "Edit class assumptions and adjustment multipliers directly in the source CSVs."
    )

    class_config = load_class_config()
    adjustment_config = load_adjustment_config()

    st.subheader("Class Config")
    st.write("Editable fields: base class weight and base expected hours.")
    class_editor_df = build_class_config_editor_df(class_config)
    edited_class_df = st.data_editor(
        class_editor_df,
        hide_index=True,
        num_rows="fixed",
        disabled=["task_class", "class_name", "is_active", "updated_at"],
        key="class_config_editor",
        use_container_width=True,
    )
    try:
        validated_class_df = validate_class_config_edit(edited_class_df)
        class_changes = get_class_config_changes(class_config, validated_class_df)
    except ValueError as exc:
        validated_class_df = None
        class_changes = pd.DataFrame()
        st.error(str(exc))

    st.write("Change Preview")
    if validated_class_df is None:
        st.info("Fix the class config errors above before saving.")
    elif class_changes.empty:
        st.info("No class config changes detected.")
    else:
        st.dataframe(class_changes, hide_index=True, use_container_width=True)

    if st.button("Save Class Config", use_container_width=True):
        if validated_class_df is None:
            st.error("Class config cannot be saved until validation errors are resolved.")
        else:
            saved = save_class_config(validated_class_df)
            if not saved:
                st.info("No class config changes were detected. Nothing was saved.")
            else:
                load_results.clear()
                st.success("class_config.csv saved.")
                st.rerun()

    st.divider()

    st.subheader("Adjustment Config")
    st.write("Editable field: multiplier.")
    adjustment_editor_df = build_adjustment_config_editor_df(adjustment_config)
    edited_adjustment_df = st.data_editor(
        adjustment_editor_df,
        hide_index=True,
        num_rows="fixed",
        disabled=["adjustment_flag", "label", "min_bound", "max_bound", "is_active", "updated_at"],
        key="adjustment_config_editor",
        use_container_width=True,
    )
    try:
        validated_adjustment_df = validate_adjustment_config_edit(
            edited_adjustment_df, adjustment_config
        )
        adjustment_changes = get_adjustment_config_changes(
            adjustment_config, validated_adjustment_df
        )
    except ValueError as exc:
        validated_adjustment_df = None
        adjustment_changes = pd.DataFrame()
        st.error(str(exc))

    st.write("Change Preview")
    if validated_adjustment_df is None:
        st.info("Fix the adjustment config errors above before saving.")
    elif adjustment_changes.empty:
        st.info("No adjustment config changes detected.")
    else:
        st.dataframe(adjustment_changes, hide_index=True, use_container_width=True)

    if st.button("Save Adjustment Config", use_container_width=True):
        if validated_adjustment_df is None:
            st.error("Adjustment config cannot be saved until validation errors are resolved.")
        else:
            saved = save_adjustment_config(validated_adjustment_df)
            if not saved:
                st.info("No adjustment config changes were detected. Nothing was saved.")
            else:
                load_results.clear()
                st.success("adjustment_config.csv saved.")
                st.rerun()


def main() -> None:
    results_by_intern = load_results()
    intern_ids = sorted(results_by_intern.keys())
    selected_intern_id = st.sidebar.selectbox("Selected Intern ID", intern_ids)
    selected_results = results_by_intern[selected_intern_id]
    summary = selected_results.summary
    task_metrics = selected_results.task_metrics

    page = st.sidebar.radio(
        "Navigate",
        [
            "Manager View",
            "Intern Overview",
            "Task Breakdown",
            "Flags & Diagnostics",
            "Cross-Intern Insights",
            "Diagnostic Insights",
            "Admin Controls",
        ],
    )

    if page == "Manager View":
        render_manager_view(results_by_intern, selected_intern_id)
    elif page == "Intern Overview":
        render_overview(summary)
    elif page == "Task Breakdown":
        render_task_breakdown(task_metrics)
    elif page == "Flags & Diagnostics":
        render_flags_diagnostics(summary, task_metrics, selected_results.attribution)
    elif page == "Cross-Intern Insights":
        render_cross_intern_insights(results_by_intern)
    elif page == "Diagnostic Insights":
        render_diagnostic_insights(results_by_intern, selected_intern_id)
    elif page == "Admin Controls":
        render_admin_controls()


if __name__ == "__main__":
    main()
