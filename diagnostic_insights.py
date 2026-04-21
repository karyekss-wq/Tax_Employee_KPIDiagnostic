from __future__ import annotations

from typing import Any

import pandas as pd


COMPONENT_TIE_BREAK_ORDER = [
    "output_score",
    "efficiency_score",
    "accuracy_score",
    "contribution_modifier",
]

RANK_METRICS = [
    "final_score",
    "performance_index",
    "output_score",
    "efficiency_score",
    "accuracy_score",
    "contribution_modifier",
]


def build_cross_intern_comparison(results_by_intern: dict[str, Any]) -> pd.DataFrame:
    """
    Build a cross-intern comparison table from existing summary fields only.
    """
    rows = []
    for intern_id in sorted(results_by_intern.keys()):
        summary = results_by_intern[intern_id].summary
        rows.append(
            {
                "intern_id": str(intern_id),
                "final_score": float(summary["final_score"]),
                "performance_index": float(summary["performance_index"]),
                "output_score": float(summary["output_score"]),
                "efficiency_score": float(summary["efficiency_score"]),
                "accuracy_score": float(summary["accuracy_score"]),
                "contribution_modifier": float(summary["contribution_modifier"]),
            }
        )
    return pd.DataFrame(rows)


def get_metric_peer_context(comparison_df: pd.DataFrame, intern_id: str, metric: str) -> dict[str, Any]:
    """
    Return selected value, peer mean, peer gap, and deterministic rank for one metric.
    """
    selected_row = comparison_df.loc[comparison_df["intern_id"] == intern_id].iloc[0]
    selected_value = float(selected_row[metric])
    peer_mean = float(comparison_df[metric].mean())
    peer_gap = selected_value - peer_mean

    ranked = comparison_df.sort_values(
        [metric, "intern_id"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    rank = int(ranked.index[ranked["intern_id"] == intern_id][0]) + 1

    return {
        "metric": metric,
        "selected_value": selected_value,
        "peer_mean": peer_mean,
        "peer_gap": float(peer_gap),
        "rank": rank,
        "total": int(len(comparison_df)),
    }


def _choose_component_gap_metric(gap_by_metric: dict[str, float], positive: bool) -> tuple[str, float] | None:
    if positive:
        candidates = [m for m in COMPONENT_TIE_BREAK_ORDER if gap_by_metric[m] > 0]
        if not candidates:
            return None
        best_metric = max(candidates, key=lambda m: (gap_by_metric[m], -COMPONENT_TIE_BREAK_ORDER.index(m)))
        return best_metric, gap_by_metric[best_metric]

    candidates = [m for m in COMPONENT_TIE_BREAK_ORDER if gap_by_metric[m] < 0]
    if not candidates:
        return None
    best_metric = min(candidates, key=lambda m: (gap_by_metric[m], COMPONENT_TIE_BREAK_ORDER.index(m)))
    return best_metric, gap_by_metric[best_metric]


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def _format_signed(value: float) -> str:
    return f"{value:+.4f}"


def build_intern_diagnostic_summary(results_by_intern: dict[str, Any], intern_id: str) -> dict[str, str]:
    """
    Build deterministic intern-level diagnostic summary from existing outputs.
    """
    comparison_df = build_cross_intern_comparison(results_by_intern)
    summary = results_by_intern[intern_id].summary
    attribution = results_by_intern[intern_id].attribution

    component_contexts = {
        metric: get_metric_peer_context(comparison_df, intern_id, metric)
        for metric in COMPONENT_TIE_BREAK_ORDER
    }
    component_gaps = {metric: ctx["peer_gap"] for metric, ctx in component_contexts.items()}

    strength_choice = _choose_component_gap_metric(component_gaps, positive=True)
    if strength_choice:
        strength_metric, strength_gap = strength_choice
        primary_strength_driver = (
            f"{_metric_label(strength_metric).title()} leads peer mean by "
            f"{strength_gap:.4f}, making it the strongest support for final score."
        )
    else:
        closest_metric = max(
            COMPONENT_TIE_BREAK_ORDER,
            key=lambda m: (component_gaps[m], -COMPONENT_TIE_BREAK_ORDER.index(m)),
        )
        primary_strength_driver = (
            f"No component is above peer mean; {_metric_label(closest_metric)} is closest at "
            f"{_format_signed(component_gaps[closest_metric])}."
        )

    weakness_choice = _choose_component_gap_metric(component_gaps, positive=False)
    if weakness_choice:
        weakness_metric, weakness_gap = weakness_choice
        primary_weakness_driver = (
            f"{_metric_label(weakness_metric).title()} trails peer mean by "
            f"{abs(weakness_gap):.4f}, making it the largest drag on final score."
        )
    else:
        primary_weakness_driver = "No component is below peer mean."

    execution_components = ["efficiency_score", "accuracy_score", "contribution_modifier"]
    weakest_component = min(
        execution_components,
        key=lambda m: (float(summary[m]), execution_components.index(m)),
    )

    dominant_final_score_driver = (
        f"Final score is most constrained by {weakest_component} at {float(summary[weakest_component]):.4f}."
    )

    if weakest_component == "accuracy_score":
        accuracy_attr = attribution["accuracy_attribution"]
        by_severity = accuracy_attr.get("by_severity", [])
        severity_map = {row.get("severity"): float(row.get("weighted_error_impact", 0.0)) for row in by_severity}
        major_impact = severity_map.get("major", 0.0)
        minor_impact = severity_map.get("minor", 0.0)
        top_error_drivers = accuracy_attr.get("top_error_drivers", [])
        if major_impact > minor_impact:
            if top_error_drivers:
                top_task = top_error_drivers[0]
                dominant_final_score_driver = (
                    "Final score is most constrained by accuracy_score, driven primarily by major-error "
                    f"impact led by task {top_task['task_id']}."
                )
            else:
                dominant_final_score_driver = (
                    "Final score is most constrained by accuracy_score, with major errors driving more "
                    "weighted impact than minor errors."
                )
    elif weakest_component == "efficiency_score":
        efficiency_attr = attribution["efficiency_attribution"]
        overruns = efficiency_attr.get("largest_overruns", [])
        if overruns:
            top_overrun = overruns[0]
            dominant_final_score_driver = (
                "Final score is most constrained by efficiency_score, with the largest overrun from "
                f"task {top_overrun['task_id']} ({top_overrun['task_class']})."
            )
    elif weakest_component == "contribution_modifier":
        contribution_attr = attribution["contribution_attribution"]
        if int(summary["negative_flags"]) > int(summary["positive_flags"]):
            negative_by_type = contribution_attr.get("negative_by_type", [])
            if negative_by_type:
                top_flag = negative_by_type[0]
                dominant_final_score_driver = (
                    "Final score is most constrained by contribution_modifier, reduced by negative flags "
                    f"concentrated in {top_flag['flag_type']}."
                )

    performance_index = float(summary["performance_index"])
    if performance_index >= 0.90:
        profile_label = "strong execution profile"
    elif performance_index >= 0.75:
        profile_label = "stable but mixed execution profile"
    else:
        profile_label = "execution risk profile"

    lowest_execution_component = min(
        execution_components,
        key=lambda m: (float(summary[m]), execution_components.index(m)),
    )
    performance_index_interpretation = (
        f"Performance index ({performance_index:.4f}) indicates a {profile_label}; "
        f"lowest execution component is {lowest_execution_component} "
        f"at {float(summary[lowest_execution_component]):.4f}."
    )

    return {
        "primary_strength_driver": primary_strength_driver,
        "primary_weakness_driver": primary_weakness_driver,
        "dominant_final_score_driver": dominant_final_score_driver,
        "performance_index_interpretation": performance_index_interpretation,
    }


def build_cross_intern_positioning(results_by_intern: dict[str, Any], intern_id: str) -> dict[str, Any]:
    """
    Build rank-based positioning and peer-gap narrative from existing summaries.
    """
    comparison_df = build_cross_intern_comparison(results_by_intern)
    contexts = {
        metric: get_metric_peer_context(comparison_df, intern_id, metric)
        for metric in RANK_METRICS
    }

    component_gaps = {
        metric: contexts[metric]["peer_gap"] for metric in COMPONENT_TIE_BREAK_ORDER
    }
    support_choice = _choose_component_gap_metric(component_gaps, positive=True)
    drag_choice = _choose_component_gap_metric(component_gaps, positive=False)

    if support_choice:
        support_metric, support_gap = support_choice
        support_text = f"above-peer {_metric_label(support_metric)} ({_format_signed(support_gap)})"
    else:
        support_text = "no above-peer execution component"

    if drag_choice:
        drag_metric, drag_gap = drag_choice
        drag_text = f"below-peer {_metric_label(drag_metric)} ({_format_signed(drag_gap)})"
    else:
        drag_text = "no below-peer execution component"

    final_ctx = contexts["final_score"]
    perf_ctx = contexts["performance_index"]

    final_score_positioning = (
        f"Ranks {final_ctx['rank']} of {final_ctx['total']} on final_score, "
        f"supported by {support_text} and constrained by {drag_text}."
    )

    if drag_choice:
        performance_index_positioning = (
            f"Ranks {perf_ctx['rank']} of {perf_ctx['total']} on performance_index because "
            f"{drag_text}."
        )
    else:
        performance_index_positioning = (
            f"Ranks {perf_ctx['rank']} of {perf_ctx['total']} on performance_index with no "
            "execution component below peer mean."
        )

    positive_metric_candidates = [m for m in RANK_METRICS if contexts[m]["peer_gap"] > 0]
    if positive_metric_candidates:
        strongest_peer_advantage_metric = max(
            positive_metric_candidates,
            key=lambda m: (contexts[m]["peer_gap"], -RANK_METRICS.index(m)),
        )
        strongest_peer_advantage = (
            f"Strongest peer advantage: {_metric_label(strongest_peer_advantage_metric)} "
            f"{_format_signed(contexts[strongest_peer_advantage_metric]['peer_gap'])}."
        )
    else:
        strongest_peer_advantage = "No metric is above peer mean."

    largest_gap_metric = max(
        RANK_METRICS,
        key=lambda m: (abs(contexts[m]["peer_gap"]), -RANK_METRICS.index(m)),
    )
    largest_gap_value = contexts[largest_gap_metric]["peer_gap"]
    direction = "above" if largest_gap_value > 0 else "below"
    largest_peer_gap = (
        f"Largest peer gap: {_metric_label(largest_gap_metric)} is {direction} peer mean by "
        f"{abs(largest_gap_value):.4f}."
    )

    peer_comparison_highlights = [
        f"Final score rank: {final_ctx['rank']}/{final_ctx['total']} ({_format_signed(final_ctx['peer_gap'])} vs peer mean).",
        f"Performance index rank: {perf_ctx['rank']}/{perf_ctx['total']} ({_format_signed(perf_ctx['peer_gap'])} vs peer mean).",
        largest_peer_gap,
    ]

    return {
        "final_score_rank": final_ctx["rank"],
        "final_score_total": final_ctx["total"],
        "performance_index_rank": perf_ctx["rank"],
        "performance_index_total": perf_ctx["total"],
        "final_score_positioning": final_score_positioning,
        "performance_index_positioning": performance_index_positioning,
        "peer_comparison_highlights": peer_comparison_highlights,
        "strongest_peer_advantage": strongest_peer_advantage,
        "largest_peer_gap": largest_peer_gap,
    }


def build_attribution_explanations(results_by_intern: dict[str, Any], intern_id: str) -> dict[str, str]:
    """
    Build deterministic explanation sentences using attribution payload only.
    """
    results = results_by_intern[intern_id]
    summary = results.summary
    attribution = results.attribution

    output_attr = attribution["output_attribution"]
    output_by_class = output_attr.get("by_class", [])
    output_by_adjustment = output_attr.get("by_adjustment", [])
    top_positive_adjustment = next(
        (row for row in output_by_adjustment if float(row.get("output_effect", 0.0)) > 0),
        None,
    )
    if top_positive_adjustment:
        output_explanation = (
            "Output is lifted by adjustment-heavy workload, led by "
            f"{top_positive_adjustment['adjustment_code']} "
            f"(effect {float(top_positive_adjustment['output_effect']):.4f})."
        )
    elif output_by_class:
        top_class = output_by_class[0]
        output_explanation = (
            "Output is concentrated in "
            f"{top_class['task_class']} ({top_class['class_name']}) with "
            f"{float(top_class['output_contribution']):.4f} contribution."
        )
    else:
        output_explanation = "Output attribution has no class or adjustment drivers to report."

    efficiency_attr = attribution["efficiency_attribution"]
    overruns = efficiency_attr.get("largest_overruns", [])
    underruns = efficiency_attr.get("largest_underruns", [])
    top_overrun = overruns[0] if overruns else None
    top_underrun = underruns[0] if underruns else None

    if top_overrun and float(top_overrun.get("overrun_hours", 0.0)) > 0:
        efficiency_explanation = (
            "Efficiency is suppressed by overruns concentrated in "
            f"{top_overrun['task_class']} tasks, led by {top_overrun['task_id']} "
            f"({float(top_overrun['overrun_hours']):.4f} overrun hours)."
        )
    elif top_underrun and float(top_underrun.get("underrun_hours", 0.0)) > 0:
        efficiency_explanation = (
            "Efficiency is supported by underruns, led by "
            f"{top_underrun['task_id']} ({float(top_underrun['underrun_hours']):.4f} underrun hours)."
        )
    else:
        efficiency_explanation = "Efficiency attribution shows no concentrated overrun or underrun driver."

    accuracy_attr = attribution["accuracy_attribution"]
    top_error_drivers = accuracy_attr.get("top_error_drivers", [])
    by_severity = accuracy_attr.get("by_severity", [])
    severity_map = {row.get("severity"): float(row.get("weighted_error_impact", 0.0)) for row in by_severity}
    major_impact = severity_map.get("major", 0.0)
    minor_impact = severity_map.get("minor", 0.0)

    if top_error_drivers:
        top_error_task = top_error_drivers[0]
        if major_impact > minor_impact:
            accuracy_explanation = (
                "Accuracy drag is driven by major-error concentration, led by task "
                f"{top_error_task['task_id']}."
            )
        else:
            accuracy_explanation = (
                "Accuracy drag is concentrated in task "
                f"{top_error_task['task_id']} ({top_error_task['task_class']}) with "
                f"{float(top_error_task['weighted_errors']):.4f} weighted errors."
            )
    elif float(summary["total_weighted_errors"]) == 0:
        accuracy_explanation = "Accuracy attribution shows no weighted error burden."
    else:
        accuracy_explanation = "Accuracy attribution has weighted error burden without a concentrated task driver."

    contribution_attr = attribution["contribution_attribution"]
    negative_by_type = contribution_attr.get("negative_by_type", [])
    positive_by_type = contribution_attr.get("positive_by_type", [])
    raw_negative_effect = float(contribution_attr.get("raw_negative_effect", 0.0))
    raw_positive_effect = float(contribution_attr.get("raw_positive_effect", 0.0))

    if raw_negative_effect < 0 and negative_by_type:
        top_negative = negative_by_type[0]
        contribution_explanation = (
            "Contribution is reduced by negative flag concentration in "
            f"{top_negative['flag_type']} "
            f"({abs(float(top_negative['modifier_effect'])):.4f} modifier impact)."
        )
    elif raw_positive_effect > 0 and positive_by_type:
        top_positive = positive_by_type[0]
        contribution_explanation = (
            "Contribution is supported by positive flag concentration in "
            f"{top_positive['flag_type']} "
            f"({float(top_positive['modifier_effect']):.4f} modifier impact)."
        )
    else:
        contribution_explanation = "Contribution attribution shows no dominant positive or negative flag concentration."

    return {
        "output_explanation": output_explanation,
        "efficiency_explanation": efficiency_explanation,
        "accuracy_explanation": accuracy_explanation,
        "contribution_explanation": contribution_explanation,
    }


def build_diagnostic_insights(results_by_intern: dict[str, Any], intern_id: str) -> dict[str, Any]:
    """
    Build full deterministic diagnostic interpretation payload for one intern.
    """
    return {
        "intern_summary": build_intern_diagnostic_summary(results_by_intern, intern_id),
        "cross_intern_positioning": build_cross_intern_positioning(results_by_intern, intern_id),
        "attribution_explanations": build_attribution_explanations(results_by_intern, intern_id),
    }
