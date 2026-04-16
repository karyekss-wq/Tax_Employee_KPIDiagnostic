from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from scoring import ScoreResults, run_scoring


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
            "Intern Overview",
            "Task Breakdown",
            "Flags & Diagnostics",
            "Admin Controls",
        ],
    )

    if page == "Intern Overview":
        render_overview(summary)
    elif page == "Task Breakdown":
        render_task_breakdown(task_metrics)
    elif page == "Flags & Diagnostics":
        render_flags_diagnostics(summary, task_metrics, selected_results.attribution)
    elif page == "Admin Controls":
        render_admin_controls()


if __name__ == "__main__":
    main()
