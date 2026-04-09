from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from scoring import run_scoring


BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"


st.set_page_config(
    page_title="Tax Intern KPI Dashboard",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_results():
    """
    Load scored MVP results once per session unless files change.
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


def validate_class_config_edit(edited_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the editable class config slice before persisting.
    """
    expected_cols = ["task_class", "base_class_weight", "base_expected_hours"]
    if edited_df.columns.tolist() != expected_cols:
        raise ValueError(
            "class_config editor columns must remain exactly: "
            "task_class, base_class_weight, base_expected_hours"
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


def validate_adjustment_config_edit(edited_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the editable adjustment config slice before persisting.
    """
    expected_cols = ["adjustment_flag", "multiplier"]
    if edited_df.columns.tolist() != expected_cols:
        raise ValueError(
            "adjustment_config editor columns must remain exactly: "
            "adjustment_flag, multiplier"
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

    return validated


def save_class_config(edited_df: pd.DataFrame) -> None:
    """
    Persist validated class config edits back to the existing CSV.
    """
    validated = validate_class_config_edit(edited_df)
    class_config = load_class_config()
    updated = class_config.copy()

    current_values = class_config[
        ["task_class", "base_class_weight", "base_expected_hours"]
    ].reset_index(drop=True)
    changed_mask = (
        current_values.set_index("task_class")
        != validated.set_index("task_class")
    ).any(axis=1)

    updated = updated.merge(validated, on="task_class", how="left", suffixes=("", "_edited"))
    updated["base_class_weight"] = updated["base_class_weight_edited"]
    updated["base_expected_hours"] = updated["base_expected_hours_edited"]
    updated = updated.drop(
        columns=["base_class_weight_edited", "base_expected_hours_edited"]
    )

    if "updated_at" in updated.columns and changed_mask.any():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated.loc[
            updated["task_class"].isin(changed_mask.index[changed_mask]), "updated_at"
        ] = timestamp

    updated.to_csv(CONFIG_DIR / "class_config.csv", index=False)


def save_adjustment_config(edited_df: pd.DataFrame) -> None:
    """
    Persist validated adjustment config edits back to the existing CSV.
    """
    validated = validate_adjustment_config_edit(edited_df)
    adjustment_config = load_adjustment_config()
    updated = adjustment_config.copy()

    current_values = adjustment_config[
        ["adjustment_code", "multiplier_add"]
    ].rename(
        columns={
            "adjustment_code": "adjustment_flag",
            "multiplier_add": "multiplier",
        }
    ).reset_index(drop=True)
    changed_mask = (
        current_values.set_index("adjustment_flag")
        != validated.set_index("adjustment_flag")
    ).any(axis=1)

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

    if "updated_at" in updated.columns and changed_mask.any():
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        updated.loc[
            updated["adjustment_code"].isin(changed_mask.index[changed_mask]), "updated_at"
        ] = timestamp

    updated.to_csv(CONFIG_DIR / "adjustment_config.csv", index=False)


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


def render_overview(summary: dict) -> None:
    st.title("First-Year Tax Intern Performance Dashboard")
    st.caption("Busy season MVP demo for a single employee using mock data.")

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


def render_flags_diagnostics(summary: dict, task_metrics: pd.DataFrame) -> None:
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


def render_admin_controls() -> None:
    st.header("Admin Controls")
    st.caption(
        "Edit class assumptions and adjustment multipliers directly in the source CSVs."
    )

    class_config = load_class_config()
    adjustment_config = load_adjustment_config()

    st.subheader("Class Config")
    st.write("Editable fields: base class weight and base expected hours.")
    class_editor_df = class_config[
        ["task_class", "base_class_weight", "base_expected_hours"]
    ].copy()
    edited_class_df = st.data_editor(
        class_editor_df,
        hide_index=True,
        num_rows="fixed",
        disabled=["task_class"],
        key="class_config_editor",
        use_container_width=True,
    )
    if st.button("Save Class Config", use_container_width=True):
        try:
            save_class_config(edited_class_df)
        except ValueError as exc:
            st.error(str(exc))
        else:
            load_results.clear()
            st.success("class_config.csv saved.")
            st.rerun()

    st.divider()

    st.subheader("Adjustment Config")
    st.write("Editable field: multiplier.")
    adjustment_editor_df = adjustment_config[
        ["adjustment_code", "multiplier_add"]
    ].rename(
        columns={
            "adjustment_code": "adjustment_flag",
            "multiplier_add": "multiplier",
        }
    )
    edited_adjustment_df = st.data_editor(
        adjustment_editor_df,
        hide_index=True,
        num_rows="fixed",
        disabled=["adjustment_flag"],
        key="adjustment_config_editor",
        use_container_width=True,
    )
    if st.button("Save Adjustment Config", use_container_width=True):
        try:
            save_adjustment_config(edited_adjustment_df)
        except ValueError as exc:
            st.error(str(exc))
        else:
            load_results.clear()
            st.success("adjustment_config.csv saved.")
            st.rerun()


def main() -> None:
    results = load_results()
    summary = results.summary
    task_metrics = results.task_metrics

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
        render_flags_diagnostics(summary, task_metrics)
    elif page == "Admin Controls":
        render_admin_controls()


if __name__ == "__main__":
    main()
