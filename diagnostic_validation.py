from __future__ import annotations

import re
from typing import Any

from diagnostic_insights import (
    COMPONENT_TIE_BREAK_ORDER,
    RANK_METRICS,
    build_cross_intern_comparison,
    build_diagnostic_insights,
    get_metric_peer_context,
)


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def _format_signed(value: float) -> str:
    return f"{value:+.4f}"


def _choose_component_gap_metric(gap_by_metric: dict[str, float], positive: bool) -> tuple[str, float] | None:
    if positive:
        candidates = [m for m in COMPONENT_TIE_BREAK_ORDER if gap_by_metric[m] > 0]
        if not candidates:
            return None
        best = max(candidates, key=lambda m: (gap_by_metric[m], -COMPONENT_TIE_BREAK_ORDER.index(m)))
        return best, gap_by_metric[best]

    candidates = [m for m in COMPONENT_TIE_BREAK_ORDER if gap_by_metric[m] < 0]
    if not candidates:
        return None
    best = min(candidates, key=lambda m: (gap_by_metric[m], COMPONENT_TIE_BREAK_ORDER.index(m)))
    return best, gap_by_metric[best]


def _lowest_execution_component(summary: dict[str, Any]) -> str:
    execution = ["efficiency_score", "accuracy_score", "contribution_modifier"]
    return min(execution, key=lambda m: (float(summary[m]), execution.index(m)))


def _get_top_positive_adjustment(attribution: dict[str, Any]) -> dict[str, Any] | None:
    by_adjustment = attribution["output_attribution"].get("by_adjustment", [])
    return next((row for row in by_adjustment if float(row.get("output_effect", 0.0)) > 0), None)


def validate_diagnostic_insights(results_by_intern: dict[str, Any]) -> list[str]:
    """
    Validate deterministic consistency of diagnostic insights against source outputs.

    Returns:
        list of human-readable validation errors. Empty list means success.
    """
    errors: list[str] = []
    comparison_df = build_cross_intern_comparison(results_by_intern)

    for intern_id in sorted(results_by_intern.keys()):
        result = results_by_intern[intern_id]
        summary = result.summary
        attribution = result.attribution
        insights = build_diagnostic_insights(results_by_intern, intern_id)

        if sorted(insights.keys()) != [
            "attribution_explanations",
            "cross_intern_positioning",
            "intern_summary",
            "normalized_insights",
        ]:
            errors.append(f"{intern_id} top-level keys mismatch: {sorted(insights.keys())}")
            continue

        intern_summary = insights["intern_summary"]
        positioning = insights["cross_intern_positioning"]
        attr_exp = insights["attribution_explanations"]
        normalized_insights = insights["normalized_insights"]

        required_normalized_fields = [
            "intern_id",
            "insight_key",
            "insight_type",
            "metric_source",
            "direction",
            "severity",
            "evidence_value",
            "evidence_unit",
            "evidence_label",
            "supporting_reference",
            "message",
        ]
        if not isinstance(normalized_insights, list):
            errors.append(f"{intern_id} normalized_insights must be a list")
        else:
            for idx, record in enumerate(normalized_insights):
                if not isinstance(record, dict):
                    errors.append(f"{intern_id} normalized_insights[{idx}] must be a dict")
                    continue
                for field in required_normalized_fields:
                    if field not in record:
                        errors.append(
                            f"{intern_id} normalized_insights[{idx}] missing required field '{field}'"
                        )

        for key in [
            "primary_strength_driver",
            "primary_weakness_driver",
            "dominant_final_score_driver",
            "performance_index_interpretation",
        ]:
            if key not in intern_summary:
                errors.append(f"{intern_id} missing intern_summary key: {key}")

        for key in [
            "final_score_rank",
            "final_score_total",
            "performance_index_rank",
            "performance_index_total",
            "final_score_positioning",
            "performance_index_positioning",
            "peer_comparison_highlights",
            "strongest_peer_advantage",
            "largest_peer_gap",
        ]:
            if key not in positioning:
                errors.append(f"{intern_id} missing cross_intern_positioning key: {key}")

        for key in [
            "output_explanation",
            "efficiency_explanation",
            "accuracy_explanation",
            "contribution_explanation",
        ]:
            if key not in attr_exp:
                errors.append(f"{intern_id} missing attribution_explanations key: {key}")

        contexts = {
            metric: get_metric_peer_context(comparison_df, intern_id, metric)
            for metric in RANK_METRICS
        }
        component_gaps = {metric: contexts[metric]["peer_gap"] for metric in COMPONENT_TIE_BREAK_ORDER}

        # A. Intern summary consistency
        expected_strength = _choose_component_gap_metric(component_gaps, positive=True)
        strength_text = intern_summary["primary_strength_driver"]
        if expected_strength:
            metric, gap = expected_strength
            if _metric_label(metric).title() not in strength_text or f"{gap:.4f}" not in strength_text:
                errors.append(
                    f"{intern_id} strength driver mismatch: expected {metric} ({gap:.4f}), "
                    f"found '{strength_text}'"
                )
        else:
            closest_metric = max(
                COMPONENT_TIE_BREAK_ORDER,
                key=lambda m: (component_gaps[m], -COMPONENT_TIE_BREAK_ORDER.index(m)),
            )
            expected_fragment = f"No component is above peer mean; {_metric_label(closest_metric)}"
            if expected_fragment not in strength_text:
                errors.append(
                    f"{intern_id} strength fallback mismatch: expected fragment '{expected_fragment}', "
                    f"found '{strength_text}'"
                )

        expected_weakness = _choose_component_gap_metric(component_gaps, positive=False)
        weakness_text = intern_summary["primary_weakness_driver"]
        if expected_weakness:
            metric, gap = expected_weakness
            if _metric_label(metric).title() not in weakness_text or f"{abs(gap):.4f}" not in weakness_text:
                errors.append(
                    f"{intern_id} weakness driver mismatch: expected {metric} ({gap:.4f}), "
                    f"found '{weakness_text}'"
                )
        elif weakness_text != "No component is below peer mean.":
            errors.append(
                f"{intern_id} weakness fallback mismatch: expected 'No component is below peer mean.', "
                f"found '{weakness_text}'"
            )

        dominant_text = intern_summary["dominant_final_score_driver"]
        weakest_component = _lowest_execution_component(summary)
        if weakest_component == "accuracy_score":
            by_severity = attribution["accuracy_attribution"].get("by_severity", [])
            severity = {row.get("severity"): float(row.get("weighted_error_impact", 0.0)) for row in by_severity}
            major = severity.get("major", 0.0)
            minor = severity.get("minor", 0.0)
            top_error_drivers = attribution["accuracy_attribution"].get("top_error_drivers", [])
            if major > minor:
                if top_error_drivers:
                    task_id = str(top_error_drivers[0]["task_id"])
                    if task_id not in dominant_text or "major-error" not in dominant_text:
                        errors.append(
                            f"{intern_id} dominant driver mismatch: expected major-error text with task {task_id}, "
                            f"found '{dominant_text}'"
                        )
                elif "major errors driving more weighted impact" not in dominant_text:
                    errors.append(
                        f"{intern_id} dominant driver mismatch: expected major-impact fallback, "
                        f"found '{dominant_text}'"
                    )
            else:
                if f"{weakest_component}" not in dominant_text:
                    errors.append(
                        f"{intern_id} dominant driver mismatch: expected weakest component {weakest_component}, "
                        f"found '{dominant_text}'"
                    )
        elif weakest_component == "efficiency_score":
            overruns = attribution["efficiency_attribution"].get("largest_overruns", [])
            if overruns:
                top_overrun = overruns[0]
                if str(top_overrun["task_id"]) not in dominant_text or str(top_overrun["task_class"]) not in dominant_text:
                    errors.append(
                        f"{intern_id} dominant driver mismatch: expected top overrun task/class in '{dominant_text}'"
                    )
            elif weakest_component not in dominant_text:
                errors.append(
                    f"{intern_id} dominant driver mismatch: expected weakest component {weakest_component}, "
                    f"found '{dominant_text}'"
                )
        elif weakest_component == "contribution_modifier":
            if int(summary["negative_flags"]) > int(summary["positive_flags"]):
                neg_by_type = attribution["contribution_attribution"].get("negative_by_type", [])
                if neg_by_type:
                    flag_type = str(neg_by_type[0]["flag_type"])
                    if flag_type not in dominant_text:
                        errors.append(
                            f"{intern_id} dominant driver mismatch: expected flag_type {flag_type} in '{dominant_text}'"
                        )
                elif weakest_component not in dominant_text:
                    errors.append(
                        f"{intern_id} dominant driver mismatch: expected weakest component {weakest_component}, "
                        f"found '{dominant_text}'"
                    )
            elif weakest_component not in dominant_text:
                errors.append(
                    f"{intern_id} dominant driver mismatch: expected weakest component {weakest_component}, "
                    f"found '{dominant_text}'"
                )

        pi = float(summary["performance_index"])
        if pi >= 0.90:
            expected_band = "strong execution profile"
        elif pi >= 0.75:
            expected_band = "stable but mixed execution profile"
        else:
            expected_band = "execution risk profile"
        perf_text = intern_summary["performance_index_interpretation"]
        if expected_band not in perf_text:
            errors.append(
                f"{intern_id} performance index interpretation mismatch: expected band '{expected_band}', "
                f"found '{perf_text}'"
            )
        if _lowest_execution_component(summary) not in perf_text:
            errors.append(
                f"{intern_id} performance index interpretation missing weakest component: '{perf_text}'"
            )

        # B. Cross-intern positioning consistency
        final_ctx = contexts["final_score"]
        perf_ctx = contexts["performance_index"]
        if positioning["final_score_rank"] != final_ctx["rank"]:
            errors.append(
                f"{intern_id} rank mismatch: expected final_score rank {final_ctx['rank']}, "
                f"found {positioning['final_score_rank']}"
            )
        if positioning["performance_index_rank"] != perf_ctx["rank"]:
            errors.append(
                f"{intern_id} rank mismatch: expected performance_index rank {perf_ctx['rank']}, "
                f"found {positioning['performance_index_rank']}"
            )

        if f"Ranks {final_ctx['rank']} of {final_ctx['total']} on final_score" not in positioning[
            "final_score_positioning"
        ]:
            errors.append(
                f"{intern_id} final_score_positioning mismatch: '{positioning['final_score_positioning']}'"
            )
        if f"Ranks {perf_ctx['rank']} of {perf_ctx['total']} on performance_index" not in positioning[
            "performance_index_positioning"
        ]:
            errors.append(
                f"{intern_id} performance_index_positioning mismatch: '{positioning['performance_index_positioning']}'"
            )

        support_choice = _choose_component_gap_metric(component_gaps, positive=True)
        if support_choice:
            support_metric, support_gap = support_choice
            support_fragment = f"above-peer {_metric_label(support_metric)} ({_format_signed(support_gap)})"
            if support_fragment not in positioning["final_score_positioning"]:
                errors.append(
                    f"{intern_id} support fragment mismatch: expected '{support_fragment}' in "
                    f"'{positioning['final_score_positioning']}'"
                )
        elif "no above-peer execution component" not in positioning["final_score_positioning"]:
            errors.append(
                f"{intern_id} support fallback mismatch: '{positioning['final_score_positioning']}'"
            )

        drag_choice = _choose_component_gap_metric(component_gaps, positive=False)
        if drag_choice:
            drag_metric, drag_gap = drag_choice
            drag_fragment = f"below-peer {_metric_label(drag_metric)} ({_format_signed(drag_gap)})"
            if drag_fragment not in positioning["final_score_positioning"]:
                errors.append(
                    f"{intern_id} drag fragment mismatch: expected '{drag_fragment}' in "
                    f"'{positioning['final_score_positioning']}'"
                )
        elif "no below-peer execution component" not in positioning["final_score_positioning"]:
            errors.append(
                f"{intern_id} drag fallback mismatch: '{positioning['final_score_positioning']}'"
            )

        positive_metrics = [m for m in RANK_METRICS if contexts[m]["peer_gap"] > 0]
        if positive_metrics:
            exp_metric = max(
                positive_metrics,
                key=lambda m: (contexts[m]["peer_gap"], -RANK_METRICS.index(m)),
            )
            exp_gap = contexts[exp_metric]["peer_gap"]
            fragment = f"{_metric_label(exp_metric)} {_format_signed(exp_gap)}"
            if fragment not in positioning["strongest_peer_advantage"]:
                errors.append(
                    f"{intern_id} strongest_peer_advantage mismatch: expected '{fragment}', "
                    f"found '{positioning['strongest_peer_advantage']}'"
                )
        elif positioning["strongest_peer_advantage"] != "No metric is above peer mean.":
            errors.append(
                f"{intern_id} strongest_peer_advantage fallback mismatch: "
                f"'{positioning['strongest_peer_advantage']}'"
            )

        largest_metric = max(
            RANK_METRICS,
            key=lambda m: (abs(contexts[m]["peer_gap"]), -RANK_METRICS.index(m)),
        )
        largest_gap = contexts[largest_metric]["peer_gap"]
        direction = "above" if largest_gap > 0 else "below"
        expected_largest = (
            f"Largest peer gap: {_metric_label(largest_metric)} is {direction} peer mean by "
            f"{abs(largest_gap):.4f}."
        )
        if positioning["largest_peer_gap"] != expected_largest:
            errors.append(
                f"{intern_id} largest_peer_gap mismatch: expected '{expected_largest}', "
                f"found '{positioning['largest_peer_gap']}'"
            )

        highlights = positioning["peer_comparison_highlights"]
        if not isinstance(highlights, list) or len(highlights) < 3:
            errors.append(f"{intern_id} peer_comparison_highlights invalid: {highlights}")
        else:
            expected_h1 = (
                f"Final score rank: {final_ctx['rank']}/{final_ctx['total']} "
                f"({_format_signed(final_ctx['peer_gap'])} vs peer mean)."
            )
            expected_h2 = (
                f"Performance index rank: {perf_ctx['rank']}/{perf_ctx['total']} "
                f"({_format_signed(perf_ctx['peer_gap'])} vs peer mean)."
            )
            if highlights[0] != expected_h1:
                errors.append(
                    f"{intern_id} peer highlight 1 mismatch: expected '{expected_h1}', found '{highlights[0]}'"
                )
            if highlights[1] != expected_h2:
                errors.append(
                    f"{intern_id} peer highlight 2 mismatch: expected '{expected_h2}', found '{highlights[1]}'"
                )

        # C. Attribution explanation consistency
        output_text = attr_exp["output_explanation"]
        top_positive_adjustment = _get_top_positive_adjustment(attribution)
        if top_positive_adjustment:
            code = str(top_positive_adjustment["adjustment_code"])
            effect = float(top_positive_adjustment["output_effect"])
            if code not in output_text or f"{effect:.4f}" not in output_text:
                errors.append(
                    f"{intern_id} output explanation mismatch: expected adjustment {code} ({effect:.4f}), "
                    f"found '{output_text}'"
                )
        else:
            by_class = attribution["output_attribution"].get("by_class", [])
            if by_class:
                top_class = by_class[0]
                if str(top_class["task_class"]) not in output_text:
                    errors.append(
                        f"{intern_id} output explanation mismatch: expected class {top_class['task_class']}, "
                        f"found '{output_text}'"
                    )
            elif output_text != "Output attribution has no class or adjustment drivers to report.":
                errors.append(
                    f"{intern_id} output fallback mismatch: '{output_text}'"
                )

        efficiency_text = attr_exp["efficiency_explanation"]
        overruns = attribution["efficiency_attribution"].get("largest_overruns", [])
        underruns = attribution["efficiency_attribution"].get("largest_underruns", [])
        top_overrun = overruns[0] if overruns else None
        top_underrun = underruns[0] if underruns else None
        if top_overrun and float(top_overrun.get("overrun_hours", 0.0)) > 0:
            if str(top_overrun["task_id"]) not in efficiency_text or str(top_overrun["task_class"]) not in efficiency_text:
                errors.append(
                    f"{intern_id} efficiency explanation mismatch: expected top overrun task/class in '{efficiency_text}'"
                )
        elif top_underrun and float(top_underrun.get("underrun_hours", 0.0)) > 0:
            if str(top_underrun["task_id"]) not in efficiency_text:
                errors.append(
                    f"{intern_id} efficiency explanation mismatch: expected top underrun task in '{efficiency_text}'"
                )
        elif efficiency_text != "Efficiency attribution shows no concentrated overrun or underrun driver.":
            errors.append(
                f"{intern_id} efficiency fallback mismatch: '{efficiency_text}'"
            )

        accuracy_text = attr_exp["accuracy_explanation"]
        top_error_drivers = attribution["accuracy_attribution"].get("top_error_drivers", [])
        by_severity = attribution["accuracy_attribution"].get("by_severity", [])
        severity = {row.get("severity"): float(row.get("weighted_error_impact", 0.0)) for row in by_severity}
        major = severity.get("major", 0.0)
        minor = severity.get("minor", 0.0)
        if top_error_drivers:
            top_task_id = str(top_error_drivers[0]["task_id"])
            if top_task_id not in accuracy_text:
                errors.append(
                    f"{intern_id} accuracy explanation mismatch: expected task {top_task_id} in '{accuracy_text}'"
                )
            if major > minor and "major-error" not in accuracy_text:
                errors.append(
                    f"{intern_id} accuracy explanation references major dominance incorrectly: '{accuracy_text}'"
                )
        elif float(summary["total_weighted_errors"]) == 0:
            if accuracy_text != "Accuracy attribution shows no weighted error burden.":
                errors.append(
                    f"{intern_id} accuracy fallback mismatch: '{accuracy_text}'"
                )
        elif accuracy_text != "Accuracy attribution has weighted error burden without a concentrated task driver.":
            errors.append(
                f"{intern_id} accuracy fallback mismatch: '{accuracy_text}'"
            )

        contribution_text = attr_exp["contribution_explanation"]
        contribution_attr = attribution["contribution_attribution"]
        neg_by_type = contribution_attr.get("negative_by_type", [])
        pos_by_type = contribution_attr.get("positive_by_type", [])
        raw_neg = float(contribution_attr.get("raw_negative_effect", 0.0))
        raw_pos = float(contribution_attr.get("raw_positive_effect", 0.0))
        if raw_neg < 0 and neg_by_type:
            top_flag = str(neg_by_type[0]["flag_type"])
            if top_flag not in contribution_text:
                errors.append(
                    f"{intern_id} contribution explanation mismatch: expected negative flag {top_flag}, "
                    f"found '{contribution_text}'"
                )
        elif raw_pos > 0 and pos_by_type:
            top_flag = str(pos_by_type[0]["flag_type"])
            if top_flag not in contribution_text:
                errors.append(
                    f"{intern_id} contribution explanation mismatch: expected positive flag {top_flag}, "
                    f"found '{contribution_text}'"
                )
        elif contribution_text != "Contribution attribution shows no dominant positive or negative flag concentration.":
            errors.append(
                f"{intern_id} contribution fallback mismatch: '{contribution_text}'"
            )

        # Referenced identifiers must exist.
        task_ids = set(result.task_metrics["task_id"].astype(str))
        for field_name, text in attr_exp.items():
            referenced_tasks = re.findall(r"\bT\d+\b", str(text))
            for task_id in referenced_tasks:
                if task_id not in task_ids:
                    errors.append(
                        f"{intern_id} {field_name} references unknown task_id '{task_id}'"
                    )

    return errors
