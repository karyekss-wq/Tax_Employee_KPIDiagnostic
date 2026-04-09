from __future__ import annotations

import pandas as pd
import streamlit as st

from scoring import run_scoring


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
    st.warning(
        "This MVP view is currently informational only. Editable controls should be connected "
        "to class_config.csv and adjustment_config.csv in the next iteration."
    )

    st.write("Planned editable controls:")
    st.write("- Base class weights")
    st.write("- Base expected hours")
    st.write("- Adjustment multipliers")
    st.write("- Contribution weights")
    st.write("- Category thresholds")


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
