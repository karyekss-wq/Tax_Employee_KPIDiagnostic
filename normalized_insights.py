from __future__ import annotations

from typing import Any


COMPONENT_TIE_BREAK_ORDER = [
    "output_score",
    "efficiency_score",
    "accuracy_score",
    "contribution_modifier",
]

NORMALIZED_INSIGHT_ORDER = [
    "primary_strength_driver",
    "primary_weakness_driver",
    "dominant_final_score_driver",
    "performance_index_interpretation",
    "final_score_positioning",
    "performance_index_positioning",
    "output_explanation",
    "efficiency_explanation",
    "accuracy_explanation",
    "contribution_explanation",
]


def _metric_label(metric: str) -> str:
    return metric.replace("_", " ")


def _build_comparison_context(results_by_intern: dict[str, Any], intern_id: str, metric: str) -> dict[str, Any]:
    rows = []
    for iid in sorted(results_by_intern.keys()):
        s = results_by_intern[iid].summary
        rows.append({"intern_id": str(iid), metric: float(s[metric])})

    selected = next(row for row in rows if row["intern_id"] == intern_id)
    selected_value = float(selected[metric])
    peer_mean = float(sum(float(row[metric]) for row in rows) / len(rows))
    peer_gap = float(selected_value - peer_mean)

    ranked = sorted(rows, key=lambda row: (-float(row[metric]), str(row["intern_id"])))
    rank = next(i for i, row in enumerate(ranked, start=1) if row["intern_id"] == intern_id)

    return {
        "metric": metric,
        "selected_value": selected_value,
        "peer_mean": peer_mean,
        "peer_gap": peer_gap,
        "rank": rank,
        "total": len(rows),
    }


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


def _peer_gap_severity(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude == 0:
        return "neutral"
    if magnitude < 0.05:
        return "low"
    if magnitude < 0.15:
        return "moderate"
    return "high"


def _hours_severity(hours: float) -> str:
    magnitude = abs(float(hours))
    if magnitude == 0:
        return "neutral"
    if magnitude < 1.0:
        return "low"
    if magnitude < 3.0:
        return "moderate"
    return "high"


def _impact_severity(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude == 0:
        return "neutral"
    if magnitude < 0.5:
        return "low"
    if magnitude < 1.5:
        return "moderate"
    return "high"


def _modifier_effect_severity(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude == 0:
        return "neutral"
    if magnitude < 0.02:
        return "low"
    if magnitude < 0.05:
        return "moderate"
    return "high"


def _component_score_severity(value: float) -> str:
    # Distance from neutral multiplicative baseline of 1.0.
    magnitude = abs(1.0 - float(value))
    if magnitude == 0:
        return "neutral"
    if magnitude < 0.05:
        return "low"
    if magnitude < 0.15:
        return "moderate"
    return "high"


def _normalize_intern_summary(
    *,
    intern_id: str,
    summary: dict[str, Any],
    attribution: dict[str, Any],
    intern_summary: dict[str, str],
    component_contexts: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    component_gaps = {m: component_contexts[m]["peer_gap"] for m in COMPONENT_TIE_BREAK_ORDER}

    # 1. primary_strength_driver
    strength = _choose_component_gap_metric(component_gaps, positive=True)
    if strength:
        metric, gap = strength
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "primary_strength_driver",
                "insight_type": "driver",
                "metric_source": metric,
                "direction": "strength",
                "severity": _peer_gap_severity(gap),
                "evidence_value": float(gap),
                "evidence_unit": "peer_gap",
                "evidence_label": f"Above peer mean {_metric_label(metric)}",
                "supporting_reference": "cross_intern_comparison",
                "message": intern_summary["primary_strength_driver"],
                "peer_mean": component_contexts[metric]["peer_mean"],
                "selected_value": component_contexts[metric]["selected_value"],
            }
        )
    else:
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "primary_strength_driver",
                "insight_type": "driver",
                "metric_source": "output_score",
                "direction": "neutral",
                "severity": "neutral",
                "evidence_value": 0.0,
                "evidence_unit": "peer_gap",
                "evidence_label": "No above-peer component",
                "supporting_reference": "cross_intern_comparison",
                "message": intern_summary["primary_strength_driver"],
            }
        )

    # 2. primary_weakness_driver
    weakness = _choose_component_gap_metric(component_gaps, positive=False)
    if weakness:
        metric, gap = weakness
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "primary_weakness_driver",
                "insight_type": "driver",
                "metric_source": metric,
                "direction": "weakness",
                "severity": _peer_gap_severity(gap),
                "evidence_value": float(gap),
                "evidence_unit": "peer_gap",
                "evidence_label": f"Below peer mean {_metric_label(metric)}",
                "supporting_reference": "cross_intern_comparison",
                "message": intern_summary["primary_weakness_driver"],
                "peer_mean": component_contexts[metric]["peer_mean"],
                "selected_value": component_contexts[metric]["selected_value"],
            }
        )
    else:
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "primary_weakness_driver",
                "insight_type": "driver",
                "metric_source": "output_score",
                "direction": "neutral",
                "severity": "neutral",
                "evidence_value": 0.0,
                "evidence_unit": "peer_gap",
                "evidence_label": "No below-peer component",
                "supporting_reference": "cross_intern_comparison",
                "message": intern_summary["primary_weakness_driver"],
            }
        )

    # 3. dominant_final_score_driver
    execution_components = ["efficiency_score", "accuracy_score", "contribution_modifier"]
    weakest_component = min(
        execution_components,
        key=lambda m: (float(summary[m]), execution_components.index(m)),
    )

    dominant_record: dict[str, Any] = {
        "intern_id": intern_id,
        "insight_key": "dominant_final_score_driver",
        "insight_type": "driver",
        "metric_source": weakest_component,
        "direction": "drag",
        "severity": _component_score_severity(float(summary[weakest_component])),
        "evidence_value": float(summary[weakest_component]),
        "evidence_unit": "score",
        "evidence_label": f"Lowest execution component: {weakest_component}",
        "supporting_reference": "summary",
        "message": intern_summary["dominant_final_score_driver"],
    }

    if weakest_component == "accuracy_score":
        by_severity = attribution["accuracy_attribution"].get("by_severity", [])
        severity_map = {row.get("severity"): float(row.get("weighted_error_impact", 0.0)) for row in by_severity}
        major_impact = severity_map.get("major", 0.0)
        minor_impact = severity_map.get("minor", 0.0)
        top_error_drivers = attribution["accuracy_attribution"].get("top_error_drivers", [])
        if major_impact > minor_impact:
            dominant_record.update(
                {
                    "metric_source": "accuracy_attribution",
                    "evidence_value": float(major_impact),
                    "evidence_unit": "weighted_error_impact",
                    "evidence_label": "Major error impact dominates minor impact",
                    "supporting_reference": "accuracy_attribution",
                    "severity": _impact_severity(major_impact),
                }
            )
            if top_error_drivers:
                dominant_record["related_task_id"] = str(top_error_drivers[0].get("task_id"))
    elif weakest_component == "efficiency_score":
        overruns = attribution["efficiency_attribution"].get("largest_overruns", [])
        if overruns and float(overruns[0].get("overrun_hours", 0.0)) > 0:
            top_overrun = overruns[0]
            dominant_record.update(
                {
                    "metric_source": "efficiency_attribution",
                    "evidence_value": float(top_overrun.get("overrun_hours", 0.0)),
                    "evidence_unit": "hours",
                    "evidence_label": "Largest overrun constraining efficiency",
                    "supporting_reference": "efficiency_attribution",
                    "severity": _hours_severity(float(top_overrun.get("overrun_hours", 0.0))),
                    "related_task_id": str(top_overrun.get("task_id")),
                    "related_task_class": str(top_overrun.get("task_class")),
                }
            )
    elif weakest_component == "contribution_modifier":
        if int(summary.get("negative_flags", 0)) > int(summary.get("positive_flags", 0)):
            negative_by_type = attribution["contribution_attribution"].get("negative_by_type", [])
            if negative_by_type:
                top_negative = negative_by_type[0]
                effect = abs(float(top_negative.get("modifier_effect", 0.0)))
                dominant_record.update(
                    {
                        "metric_source": "contribution_attribution",
                        "evidence_value": effect,
                        "evidence_unit": "modifier_effect",
                        "evidence_label": "Negative flag concentration reduces contribution",
                        "supporting_reference": "contribution_attribution",
                        "severity": _modifier_effect_severity(effect),
                        "related_flag_type": str(top_negative.get("flag_type")),
                    }
                )

    records.append(dominant_record)

    # 4. performance_index_interpretation
    performance_index = float(summary["performance_index"])
    if performance_index >= 0.90:
        direction = "support"
        severity = "high"
        evidence_label = "Strong execution profile band"
    elif performance_index >= 0.75:
        direction = "neutral"
        severity = "neutral"
        evidence_label = "Stable but mixed execution profile band"
    else:
        direction = "drag"
        severity = "high"
        evidence_label = "Execution risk profile band"

    records.append(
        {
            "intern_id": intern_id,
            "insight_key": "performance_index_interpretation",
            "insight_type": "interpretation",
            "metric_source": "performance_index",
            "direction": direction,
            "severity": severity,
            "evidence_value": performance_index,
            "evidence_unit": "score",
            "evidence_label": evidence_label,
            "supporting_reference": "summary",
            "message": intern_summary["performance_index_interpretation"],
            "selected_value": performance_index,
        }
    )

    return records


def _normalize_positioning(
    *,
    intern_id: str,
    positioning: dict[str, Any],
    final_ctx: dict[str, Any],
    perf_ctx: dict[str, Any],
) -> list[dict[str, Any]]:
    def direction_from_gap(gap: float) -> str:
        if gap > 0:
            return "support"
        if gap < 0:
            return "drag"
        return "neutral"

    return [
        {
            "intern_id": intern_id,
            "insight_key": "final_score_positioning",
            "insight_type": "positioning",
            "metric_source": "final_score",
            "direction": direction_from_gap(float(final_ctx["peer_gap"])),
            "severity": _peer_gap_severity(float(final_ctx["peer_gap"])),
            "evidence_value": float(final_ctx["peer_gap"]),
            "evidence_unit": "peer_gap",
            "evidence_label": "Final score versus peer mean",
            "supporting_reference": "cross_intern_comparison",
            "message": positioning["final_score_positioning"],
            "rank": int(final_ctx["rank"]),
            "peer_mean": float(final_ctx["peer_mean"]),
            "selected_value": float(final_ctx["selected_value"]),
        },
        {
            "intern_id": intern_id,
            "insight_key": "performance_index_positioning",
            "insight_type": "positioning",
            "metric_source": "performance_index",
            "direction": direction_from_gap(float(perf_ctx["peer_gap"])),
            "severity": _peer_gap_severity(float(perf_ctx["peer_gap"])),
            "evidence_value": float(perf_ctx["peer_gap"]),
            "evidence_unit": "peer_gap",
            "evidence_label": "Performance index versus peer mean",
            "supporting_reference": "cross_intern_comparison",
            "message": positioning["performance_index_positioning"],
            "rank": int(perf_ctx["rank"]),
            "peer_mean": float(perf_ctx["peer_mean"]),
            "selected_value": float(perf_ctx["selected_value"]),
        },
    ]


def _normalize_attribution_explanations(
    *,
    intern_id: str,
    summary: dict[str, Any],
    attribution: dict[str, Any],
    attribution_explanations: dict[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    # 7. output_explanation
    output_attr = attribution["output_attribution"]
    top_positive_adjustment = next(
        (row for row in output_attr.get("by_adjustment", []) if float(row.get("output_effect", 0.0)) > 0),
        None,
    )
    if top_positive_adjustment:
        effect = float(top_positive_adjustment.get("output_effect", 0.0))
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "output_explanation",
                "insight_type": "attribution",
                "metric_source": "output_attribution",
                "direction": "support",
                "severity": _impact_severity(effect),
                "evidence_value": effect,
                "evidence_unit": "output_effect",
                "evidence_label": "Top positive adjustment effect",
                "supporting_reference": "output_attribution",
                "message": attribution_explanations["output_explanation"],
                "related_adjustment_code": str(top_positive_adjustment.get("adjustment_code")),
            }
        )
    elif output_attr.get("by_class", []):
        top_class = output_attr["by_class"][0]
        contrib = float(top_class.get("output_contribution", 0.0))
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "output_explanation",
                "insight_type": "attribution",
                "metric_source": "output_attribution",
                "direction": "support" if contrib > 0 else "neutral",
                "severity": _impact_severity(contrib),
                "evidence_value": contrib,
                "evidence_unit": "output_contribution",
                "evidence_label": "Top class output contribution",
                "supporting_reference": "output_attribution",
                "message": attribution_explanations["output_explanation"],
                "related_task_class": str(top_class.get("task_class")),
            }
        )
    else:
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "output_explanation",
                "insight_type": "attribution",
                "metric_source": "output_attribution",
                "direction": "neutral",
                "severity": "neutral",
                "evidence_value": 0.0,
                "evidence_unit": "none",
                "evidence_label": "No output attribution driver",
                "supporting_reference": "output_attribution",
                "message": attribution_explanations["output_explanation"],
            }
        )

    # 8. efficiency_explanation
    efficiency_attr = attribution["efficiency_attribution"]
    overruns = efficiency_attr.get("largest_overruns", [])
    underruns = efficiency_attr.get("largest_underruns", [])
    if overruns and float(overruns[0].get("overrun_hours", 0.0)) > 0:
        top = overruns[0]
        value = float(top.get("overrun_hours", 0.0))
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "efficiency_explanation",
                "insight_type": "attribution",
                "metric_source": "efficiency_attribution",
                "direction": "drag",
                "severity": _hours_severity(value),
                "evidence_value": value,
                "evidence_unit": "hours",
                "evidence_label": "Top overrun hours",
                "supporting_reference": "efficiency_attribution",
                "message": attribution_explanations["efficiency_explanation"],
                "related_task_id": str(top.get("task_id")),
                "related_task_class": str(top.get("task_class")),
            }
        )
    elif underruns and float(underruns[0].get("underrun_hours", 0.0)) > 0:
        top = underruns[0]
        value = float(top.get("underrun_hours", 0.0))
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "efficiency_explanation",
                "insight_type": "attribution",
                "metric_source": "efficiency_attribution",
                "direction": "support",
                "severity": _hours_severity(value),
                "evidence_value": value,
                "evidence_unit": "hours",
                "evidence_label": "Top underrun hours",
                "supporting_reference": "efficiency_attribution",
                "message": attribution_explanations["efficiency_explanation"],
                "related_task_id": str(top.get("task_id")),
                "related_task_class": str(top.get("task_class")),
            }
        )
    else:
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "efficiency_explanation",
                "insight_type": "attribution",
                "metric_source": "efficiency_attribution",
                "direction": "neutral",
                "severity": "neutral",
                "evidence_value": 0.0,
                "evidence_unit": "none",
                "evidence_label": "No concentrated overrun or underrun driver",
                "supporting_reference": "efficiency_attribution",
                "message": attribution_explanations["efficiency_explanation"],
            }
        )

    # 9. accuracy_explanation
    accuracy_attr = attribution["accuracy_attribution"]
    top_error_drivers = accuracy_attr.get("top_error_drivers", [])
    severity_map = {
        row.get("severity"): float(row.get("weighted_error_impact", 0.0))
        for row in accuracy_attr.get("by_severity", [])
    }
    major_impact = severity_map.get("major", 0.0)
    minor_impact = severity_map.get("minor", 0.0)

    if top_error_drivers:
        top_task = top_error_drivers[0]
        if major_impact > minor_impact:
            records.append(
                {
                    "intern_id": intern_id,
                    "insight_key": "accuracy_explanation",
                    "insight_type": "attribution",
                    "metric_source": "accuracy_attribution",
                    "direction": "drag",
                    "severity": _impact_severity(major_impact),
                    "evidence_value": float(major_impact),
                    "evidence_unit": "weighted_error_impact",
                    "evidence_label": "Major error impact dominates",
                    "supporting_reference": "accuracy_attribution",
                    "message": attribution_explanations["accuracy_explanation"],
                    "related_task_id": str(top_task.get("task_id")),
                }
            )
        else:
            weighted_errors = float(top_task.get("weighted_errors", 0.0))
            records.append(
                {
                    "intern_id": intern_id,
                    "insight_key": "accuracy_explanation",
                    "insight_type": "attribution",
                    "metric_source": "accuracy_attribution",
                    "direction": "drag",
                    "severity": _impact_severity(weighted_errors),
                    "evidence_value": weighted_errors,
                    "evidence_unit": "weighted_errors",
                    "evidence_label": "Top weighted error driver",
                    "supporting_reference": "accuracy_attribution",
                    "message": attribution_explanations["accuracy_explanation"],
                    "related_task_id": str(top_task.get("task_id")),
                    "related_task_class": str(top_task.get("task_class")),
                }
            )
    elif float(summary.get("total_weighted_errors", 0.0)) == 0:
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "accuracy_explanation",
                "insight_type": "attribution",
                "metric_source": "accuracy_attribution",
                "direction": "support",
                "severity": "low",
                "evidence_value": 0.0,
                "evidence_unit": "weighted_errors",
                "evidence_label": "No weighted error burden",
                "supporting_reference": "accuracy_attribution",
                "message": attribution_explanations["accuracy_explanation"],
            }
        )
    else:
        total_weighted_errors = float(summary.get("total_weighted_errors", 0.0))
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "accuracy_explanation",
                "insight_type": "attribution",
                "metric_source": "accuracy_attribution",
                "direction": "drag",
                "severity": _impact_severity(total_weighted_errors),
                "evidence_value": total_weighted_errors,
                "evidence_unit": "weighted_errors",
                "evidence_label": "Weighted error burden without concentrated task",
                "supporting_reference": "accuracy_attribution",
                "message": attribution_explanations["accuracy_explanation"],
            }
        )

    # 10. contribution_explanation
    contribution_attr = attribution["contribution_attribution"]
    negative_by_type = contribution_attr.get("negative_by_type", [])
    positive_by_type = contribution_attr.get("positive_by_type", [])
    raw_negative_effect = float(contribution_attr.get("raw_negative_effect", 0.0))
    raw_positive_effect = float(contribution_attr.get("raw_positive_effect", 0.0))

    if raw_negative_effect < 0 and negative_by_type:
        top = negative_by_type[0]
        effect = abs(float(top.get("modifier_effect", 0.0)))
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "contribution_explanation",
                "insight_type": "attribution",
                "metric_source": "contribution_attribution",
                "direction": "drag",
                "severity": _modifier_effect_severity(effect),
                "evidence_value": effect,
                "evidence_unit": "modifier_effect",
                "evidence_label": "Top negative flag modifier impact",
                "supporting_reference": "contribution_attribution",
                "message": attribution_explanations["contribution_explanation"],
                "related_flag_type": str(top.get("flag_type")),
            }
        )
    elif raw_positive_effect > 0 and positive_by_type:
        top = positive_by_type[0]
        effect = float(top.get("modifier_effect", 0.0))
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "contribution_explanation",
                "insight_type": "attribution",
                "metric_source": "contribution_attribution",
                "direction": "support",
                "severity": _modifier_effect_severity(effect),
                "evidence_value": effect,
                "evidence_unit": "modifier_effect",
                "evidence_label": "Top positive flag modifier impact",
                "supporting_reference": "contribution_attribution",
                "message": attribution_explanations["contribution_explanation"],
                "related_flag_type": str(top.get("flag_type")),
            }
        )
    else:
        records.append(
            {
                "intern_id": intern_id,
                "insight_key": "contribution_explanation",
                "insight_type": "attribution",
                "metric_source": "contribution_attribution",
                "direction": "neutral",
                "severity": "neutral",
                "evidence_value": 0.0,
                "evidence_unit": "none",
                "evidence_label": "No dominant positive or negative flag concentration",
                "supporting_reference": "contribution_attribution",
                "message": attribution_explanations["contribution_explanation"],
            }
        )

    return records


def build_normalized_insights(
    *,
    results_by_intern: dict[str, Any],
    intern_id: str,
    intern_summary: dict[str, str],
    cross_intern_positioning: dict[str, Any],
    attribution_explanations: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Build deterministic, machine-consumable normalized insight records.
    """
    result = results_by_intern[intern_id]
    summary = result.summary
    attribution = result.attribution

    component_contexts = {
        metric: _build_comparison_context(results_by_intern, intern_id, metric)
        for metric in COMPONENT_TIE_BREAK_ORDER
    }
    final_ctx = _build_comparison_context(results_by_intern, intern_id, "final_score")
    perf_ctx = _build_comparison_context(results_by_intern, intern_id, "performance_index")

    records: list[dict[str, Any]] = []
    records.extend(
        _normalize_intern_summary(
            intern_id=intern_id,
            summary=summary,
            attribution=attribution,
            intern_summary=intern_summary,
            component_contexts=component_contexts,
        )
    )
    records.extend(
        _normalize_positioning(
            intern_id=intern_id,
            positioning=cross_intern_positioning,
            final_ctx=final_ctx,
            perf_ctx=perf_ctx,
        )
    )
    records.extend(
        _normalize_attribution_explanations(
            intern_id=intern_id,
            summary=summary,
            attribution=attribution,
            attribution_explanations=attribution_explanations,
        )
    )

    # Enforce stable deterministic order.
    order_index = {key: i for i, key in enumerate(NORMALIZED_INSIGHT_ORDER)}
    records = sorted(records, key=lambda row: order_index.get(row["insight_key"], 999))
    return records
